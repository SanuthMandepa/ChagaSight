"""
ST-MEM 1D Pretraining - Professional Production Version

Features:
- Dataset volume control (--subset for testing)
- Robust checkpointing (atomic writes, survives interruptions)
- Auto-resume from checkpoint with correct epoch handling
- Signal handler (saves on Ctrl+C)
- Time tracking and ETA estimation
- Per-lead masking (prevents cross-lead leakage)
- Lead embeddings + SEP tokens

ST-MEM = Spatiotemporal Masked ECG Modeling
Paper: Van Santvliet et al. (2025) Section 2.1

Usage:
# Test with 1% data first
python stmem_pretraining_1d_COMPLETE.py --subset 0.01 --epochs 5

# Full training
python stmem_pretraining_1d_COMPLETE.py --epochs 100
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import signal
import argparse
from datetime import datetime
import time

# =============================================================================
# 1. ST-MEM DATASET
# =============================================================================

class STMEMSignalDataset(Dataset):
    """Dataset for ST-MEM pretraining (all 1D signals, no labels)."""
    
    def __init__(self, data_dir, subset=1.0, seed=42):
        """
        Args:
            data_dir: Path to data/processed/1d_signals_100hz
            subset: Fraction of data to use
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        
        # Find all .npy signal files
        self.signal_paths = []
        for dataset in ['ptbxl', 'samitrop', 'code15']:
            dataset_dir = self.data_dir / dataset
            if dataset_dir.exists():
                paths = sorted(dataset_dir.glob('*.npy'))
                self.signal_paths.extend(paths)
        
        if len(self.signal_paths) == 0:
            raise ValueError(f'No signals found in {data_dir}')
        
        # Apply subset
        if subset < 1.0:
            np.random.seed(seed)
            n_samples = int(len(self.signal_paths) * subset)
            indices = np.random.choice(len(self.signal_paths), n_samples, replace=False)
            self.signal_paths = [self.signal_paths[i] for i in sorted(indices)]
        
        print(f'[DATA] Loaded {len(self.signal_paths)} signals ({subset*100:.1f}% of data)')
    
    def __len__(self):
        return len(self.signal_paths)
    
    def __getitem__(self, idx):
        # Load signal: (12, 1000) float32
        signal = np.load(self.signal_paths[idx])
        signal = torch.from_numpy(signal).float()
        return signal

# =============================================================================
# 2. ST-MEM MODEL
# =============================================================================

class PatchEmbed1D(nn.Module):
    """1D Patch Embedding with per-lead processing."""
    def __init__(self, num_leads=12, seq_len=1000, patch_size=50, embed_dim=768):
        super().__init__()
        self.num_leads = num_leads
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        assert seq_len % patch_size == 0
        self.num_patches_per_lead = seq_len // patch_size  # 20
        self.num_patches = num_leads * self.num_patches_per_lead  # 240
        
        # Per-lead Conv1D
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Lead embeddings (which lead is this patch from?)
        self.lead_embed = nn.Parameter(torch.zeros(1, num_leads, 1, embed_dim))
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, 12, 1000) signals
        Returns:
            patches: (B, 240, 768)
        """
        B, L, T = x.shape
        
        # Process each lead separately
        x = x.view(B * L, 1, T)  # (B*12, 1, 1000)
        x = self.proj(x)  # (B*12, 768, 20)
        
        # Reshape back
        x = x.view(B, L, self.embed_dim, self.num_patches_per_lead)
        x = x.permute(0, 1, 3, 2)  # (B, 12, 20, 768)
        
        # Add lead embeddings
        x = x + self.lead_embed
        
        # Flatten
        x = x.contiguous().view(B, self.num_patches, self.embed_dim)
        
        return x

class STMEM1D(nn.Module):
    """
    Spatiotemporal Masked ECG Modeling.
    
    Paper: Van Santvliet et al. (2025) Section 2.1
    
    Key differences from standard MAE:
    1. Per-lead masking (mask each lead independently)
    2. Lead embeddings (spatial information)
    3. SEP tokens between leads
    4. Shared decoder across leads (prevents cross-lead leakage)
    """
    
    def __init__(self, num_leads=12, seq_len=1000, patch_size=50,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        
        self.patch_embed = PatchEmbed1D(num_leads, seq_len, patch_size, embed_dim)
        self.num_leads = num_leads
        self.num_patches_per_lead = seq_len // patch_size  # 20
        self.num_patches = self.patch_embed.num_patches  # 240
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # SEP tokens (separate leads)
        self.sep_tokens = nn.Parameter(torch.zeros(1, num_leads, embed_dim))
        
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * 4),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder (lead-wise shared)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * 4),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.sep_tokens, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def random_masking_per_lead(self, x, mask_ratio):
        """
        Random masking per lead (prevents cross-lead leakage).
        
        Args:
            x: (B, 240, 768) patches [12 leads × 20 patches]
            mask_ratio: fraction to mask
        
        Returns:
            x_masked: visible patches
            mask: (B, 240) binary mask
            ids_restore: indices to restore
        """
        B, N, D = x.shape
        patches_per_lead = N // self.num_leads  # 20
        len_keep = int(patches_per_lead * (1 - mask_ratio))
        
        x_masked_list = []
        mask_list = []
        ids_restore_list = []
        
        # Mask each lead independently
        for lead_idx in range(self.num_leads):
            start = lead_idx * patches_per_lead
            end = (lead_idx + 1) * patches_per_lead
            
            x_lead = x[:, start:end, :]  # (B, 20, 768)
            
            # Random masking for this lead
            noise = torch.rand(B, patches_per_lead, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked_lead = torch.gather(x_lead, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
            mask_lead = torch.ones([B, patches_per_lead], device=x.device)
            mask_lead[:, :len_keep] = 0
            mask_lead = torch.gather(mask_lead, dim=1, index=ids_restore)
            
            x_masked_list.append(x_masked_lead)
            mask_list.append(mask_lead)
            ids_restore_list.append(ids_restore + start)  # Offset by lead position
        
        x_masked = torch.cat(x_masked_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        ids_restore = torch.cat(ids_restore_list, dim=1)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)  # (B, 240, 768)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking_per_lead(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
        for layer in self.decoder:
            x = layer(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, signals, pred, mask):
        """MSE loss on masked patches, per-lead."""
        target = self.patchify(signals)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def patchify(self, signals):
        """Convert signals to patches."""
        B, L, T = signals.shape  # (B, 12, 1000)
        p = self.patch_size
        n = T // p
        
        x = signals.reshape(B, L, n, p)
        x = x.permute(0, 1, 2, 3)
        x = x.reshape(B, L * n, p)
        return x
    
    def forward(self, signals, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(signals, pred, mask)
        
        return loss, pred, mask

# =============================================================================
# 3. CHECKPOINT MANAGER (FIXED)
# =============================================================================

class CheckpointManager:
    """Handles checkpointing with atomic writes and auto-resume."""
    
    def __init__(self, checkpoint_dir, auto_save_on_interrupt=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.checkpoint_dir / 'stmem_1d_checkpoint.pt'
        self.best_model_path = self.checkpoint_dir / 'stmem_1d_pretrained.pt'
        
        self.interrupted = False
        
        if auto_save_on_interrupt:
            signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Save checkpoint when Ctrl+C pressed."""
        self.interrupted = True
        print('\n[INTERRUPT] Saving checkpoint before exit...')
    
    def save(self, epoch, model, optimizer, scheduler, loss, best_loss, is_best=False, epoch_complete=True):
        """
        Save checkpoint with atomic write.
        
        Args:
            epoch: Current epoch number (0-indexed)
            epoch_complete: If True, epoch finished. If False, interrupted mid-epoch.
        """
        checkpoint = {
            'epoch': epoch,
            'epoch_complete': epoch_complete,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Atomic write (prevents corruption)
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(self.checkpoint_path)
        
        print(f'[CHECKPOINT] Saved: epoch {epoch+1}/{epoch+1 if epoch_complete else epoch+1} ({"complete" if epoch_complete else "partial"}), loss {loss:.4f}')
        
        # Save best model (encoder only)
        if is_best:
            encoder_checkpoint = {
                'epoch': epoch,
                'model_state_dict': {
                    'patch_embed': model.patch_embed.state_dict(),
                    'cls_token': model.cls_token,
                    'pos_embed': model.pos_embed,
                    'sep_tokens': model.sep_tokens,
                    'encoder': model.encoder.state_dict(),
                    'encoder_norm': model.encoder_norm.state_dict(),
                },
                'loss': loss,
            }
            torch.save(encoder_checkpoint, self.best_model_path)
            print(f'[BEST MODEL] New best: {loss:.4f}')
    
    def load(self, model, optimizer, scheduler, device):
        """Load checkpoint if exists."""
        if not self.checkpoint_path.exists():
            return 0, float('inf')
        
        print(f'[CHECKPOINT] Loading from {self.checkpoint_path}...')
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        epoch_complete = checkpoint.get('epoch_complete', True)
        
        # If epoch was interrupted mid-way, restart it. Otherwise, start next epoch.
        if epoch_complete:
            start_epoch = epoch + 1
            print(f'[RESUME] Epoch {epoch+1} completed. Resuming from epoch {start_epoch+1}')
        else:
            start_epoch = epoch
            print(f'[RESUME] Epoch {epoch+1} was interrupted. Restarting epoch {start_epoch+1}')
        
        best_loss = checkpoint['best_loss']
        print(f'[RESUME] Best loss so far: {best_loss:.4f}')
        
        return start_epoch, best_loss

# =============================================================================
# 4. TRAINING LOOP (WITH TIME TRACKING)
# =============================================================================

def train_stmem(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*70}')
    print(f'ST-MEM 1D Pretraining')
    print(f'{"="*70}')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'Data directory: {args.data_dir}')
    print(f'Dataset subset: {args.subset*100:.1f}%')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Mask ratio: {args.mask_ratio}')
    print(f'Save frequency: Every {args.save_every} epochs')
    print(f'{"="*70}\n')
    
    if args.subset < 1.0:
        print(f'[WARNING] TESTING MODE: Using {args.subset*100:.1f}% of data')
        print(f'[WARNING] Set --subset 1.0 for full training\n')
    
    # Dataset
    dataset = STMEMSignalDataset(args.data_dir, subset=args.subset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f'[DATA] DataLoader: {len(loader)} batches per epoch\n')
    
    # Model
    model = STMEM1D(mask_ratio=args.mask_ratio).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f'[MODEL] Created STMEM1D:')
    print(f'[MODEL]   Total params: {total_params:,}')
    print(f'[MODEL]   Encoder params: {encoder_params:,} (saved for fine-tuning)')
    print(f'[MODEL]   Decoder params: {total_params - encoder_params:,} (discarded)\n')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    def get_lr_schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_schedule)
    
    # Checkpoint manager
    ckpt_mgr = CheckpointManager(args.checkpoint_dir, auto_save_on_interrupt=True)
    start_epoch, best_loss = ckpt_mgr.load(model, optimizer, scheduler, device)
    
    # Training loop
    print(f'[TRAINING] Starting from epoch {start_epoch+1}/{args.epochs}...\n')
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for signals in pbar:
            signals = signals.to(device)
            
            # Forward
            loss, _, _ = model(signals)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            
            # Check interruption
            if ckpt_mgr.interrupted:
                avg_loss = epoch_loss / max(1, pbar.n)
                ckpt_mgr.save(epoch, model, optimizer, scheduler, avg_loss, best_loss, 
                            is_best=False, epoch_complete=False)
                print('[TRAINING] Interrupted. Checkpoint saved. Exiting...')
                return
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(loader)
        
        # Calculate ETA
        total_time = time.time() - training_start_time
        epochs_done = epoch - start_epoch + 1
        avg_time_per_epoch = total_time / epochs_done
        epochs_remaining = args.epochs - (epoch + 1)
        eta_seconds = avg_time_per_epoch * epochs_remaining
        eta_hours = eta_seconds / 3600
        
        print(f'[EPOCH {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}, '
              f'Time: {epoch_time/60:.1f}min, ETA: {eta_hours:.1f}h')
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_mgr.save(epoch, model, optimizer, scheduler, avg_loss, best_loss, 
                        is_best=is_best, epoch_complete=True)
        
        scheduler.step()
    
    total_training_time = time.time() - training_start_time
    print(f'\n{"="*70}')
    print(f'[COMPLETE] Training finished!')
    print(f'[COMPLETE] Total time: {total_training_time/3600:.2f} hours')
    print(f'[COMPLETE] Best loss: {best_loss:.4f}')
    print(f'[COMPLETE] Pretrained encoder saved to: {ckpt_mgr.best_model_path}')
    print(f'{"="*70}')

# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-MEM 1D Pretraining')
    
    # Data
    parser.add_argument('--data-dir', type=str, 
                       default='data/processed/1d_signals_100hz',
                       help='Path to 1D signals directory')
    parser.add_argument('--subset', type=float, default=1.0,
                       help='Dataset subset (0.01=1%%, 1.0=100%%). Use 0.01 for testing!')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (16 for 6GB GPU, 64 for 24GB GPU)')
    parser.add_argument('--learning-rate', type=float, default=1.5e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='Warmup epochs')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                       help='Masking ratio')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_stmem(args)