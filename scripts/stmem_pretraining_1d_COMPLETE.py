"""
ST-MEM 1D Pretraining - WITH BATCH-LEVEL CHECKPOINTING

KEY IMPROVEMENTS:
- Saves progress every N batches (default: 500 batches = ~15-20 min)
- Resumes from exact batch where interrupted
- No progress lost when interrupting!

Usage:
# Test with 1% data
python stmem_pretraining_1d_BATCH_CHECKPOINT.py --subset 0.01 --epochs 5

# Full training
python stmem_pretraining_1d_BATCH_CHECKPOINT.py --epochs 20 --checkpoint-every-batches 500
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
        signal = np.load(self.signal_paths[idx])
        signal = torch.from_numpy(signal).float()
        return signal

# =============================================================================
# 2. ST-MEM MODEL
# =============================================================================

class PatchEmbed1D(nn.Module):
    def __init__(self, num_leads=12, seq_len=1000, patch_size=50, embed_dim=768):
        super().__init__()
        self.num_leads = num_leads
        self.patch_size = patch_size
        self.num_patches_per_lead = seq_len // patch_size
        self.num_patches = num_leads * self.num_patches_per_lead
        
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.lead_embed = nn.Parameter(torch.zeros(1, num_leads, 1, embed_dim))
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
    
    def forward(self, x):
        B, L, T = x.shape
        x = x.view(B * L, 1, T)
        x = self.proj(x)
        x = x.view(B, L, -1, self.num_patches_per_lead).permute(0, 1, 3, 2)
        x = x + self.lead_embed
        x = x.reshape(B, self.num_patches, -1)
        return x

class STMEM1D(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, 
                 decoder_depth=4, decoder_num_heads=8, mask_ratio=0.75):
        super().__init__()
        
        self.patch_embed = PatchEmbed1D(embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_leads = self.patch_embed.num_leads
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.sep_tokens = nn.Parameter(torch.zeros(1, self.num_leads - 1, embed_dim))
        
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * 4), 
                                      dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_embed_dim, decoder_num_heads, int(decoder_embed_dim * 4),
                                      dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_embed.patch_size)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.sep_tokens, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def forward(self, signals):
        B = signals.shape[0]
        x = self.patch_embed(signals)
        
        # Add CLS + pos embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        # Random masking
        N = x.shape[1] - 1
        num_masked = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :N - num_masked]
        x_masked = torch.gather(x[:, 1:], dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x_masked = torch.cat([x[:, :1], x_masked], dim=1)
        
        # Encoder
        for layer in self.encoder:
            x_masked = layer(x_masked)
        x_masked = self.encoder_norm(x_masked)
        
        # Decoder
        x_masked = self.decoder_embed(x_masked)
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        x_full = torch.cat([x_masked[:, 1:], mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x_full = torch.cat([x_masked[:, :1], x_full], dim=1)
        x_full = x_full + self.decoder_pos_embed
        
        for layer in self.decoder:
            x_full = layer(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        
        # Compute loss
        target = signals.reshape(B, self.num_leads, self.patch_embed.num_patches_per_lead, self.patch_embed.patch_size)
        target = target.reshape(B, -1, self.patch_embed.patch_size)
        
        mask = torch.zeros(B, N, device=signals.device)
        mask.scatter_(1, ids_shuffle[:, :num_masked], 1)
        
        loss = (pred[:, 1:] - target) ** 2
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum() / self.patch_embed.patch_size
        
        return loss, pred, mask

# =============================================================================
# 3. CHECKPOINT MANAGER WITH BATCH-LEVEL SUPPORT
# =============================================================================

class BatchCheckpointManager:
    """
    Manages checkpoints at BATCH level (not just epoch level).
    
    Key features:
    - Saves every N batches
    - Resumes from exact batch where interrupted
    - No progress lost!
    """
    
    def __init__(self, checkpoint_dir, auto_save_on_interrupt=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.checkpoint_dir / 'stmem_1d_checkpoint.pt'
        self.best_model_path = self.checkpoint_dir / 'stmem_1d_pretrained.pt'
        
        self.interrupted = False
        
        if auto_save_on_interrupt:
            def signal_handler(sig, frame):
                print('\n[INTERRUPT] Ctrl+C detected. Will save checkpoint...')
                self.interrupted = True
            
            signal.signal(signal.SIGINT, signal_handler)
    
    def save(self, epoch, batch_idx, total_batches, model, optimizer, scheduler, 
             loss, best_loss, is_best=False, epoch_complete=False):
        """
        Save checkpoint with batch-level information.
        
        Args:
            epoch: Current epoch (0-indexed)
            batch_idx: Current batch within epoch (0-indexed)
            total_batches: Total batches per epoch
            epoch_complete: True if epoch finished, False if interrupted
        """
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_batches': total_batches,
            'epoch_complete': epoch_complete,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Atomic write
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(self.checkpoint_path)
        
        status = "complete" if epoch_complete else f"batch {batch_idx}/{total_batches}"
        print(f'[CHECKPOINT] Saved: epoch {epoch+1}, {status}, loss {loss:.4f}')
        
        # Save best model
        if is_best:
            encoder_checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
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
        """
        Load checkpoint if exists.
        
        Returns:
            (start_epoch, start_batch, best_loss)
        """
        if not self.checkpoint_path.exists():
            return 0, 0, float('inf')
        
        print(f'[CHECKPOINT] Loading from {self.checkpoint_path}...')
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint.get('batch_idx', 0)
        epoch_complete = checkpoint.get('epoch_complete', True)
        best_loss = checkpoint['best_loss']
        
        if epoch_complete:
            # Epoch completed - start next epoch from batch 0
            start_epoch = epoch + 1
            start_batch = 0
            print(f'[RESUME] Epoch {epoch+1} completed. Starting epoch {start_epoch+1} from batch 0')
        else:
            # Epoch interrupted - resume from next batch
            start_epoch = epoch
            start_batch = batch_idx + 1  # Resume from NEXT batch
            print(f'[RESUME] Epoch {epoch+1} interrupted at batch {batch_idx}. Resuming from batch {start_batch}')
        
        print(f'[RESUME] Best loss so far: {best_loss:.4f}')
        
        return start_epoch, start_batch, best_loss

# =============================================================================
# 4. TRAINING LOOP WITH BATCH-LEVEL CHECKPOINTING
# =============================================================================

def train_stmem(args):
    """Main training function with batch-level checkpointing."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*70}')
    print(f'ST-MEM 1D Pretraining (BATCH-LEVEL CHECKPOINTING)')
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
    print(f'Checkpoint frequency: Every {args.checkpoint_every_batches} batches')
    print(f'{"="*70}\n')
    
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
    
    total_batches = len(loader)
    print(f'[DATA] DataLoader: {total_batches} batches per epoch\n')
    
    # Model
    model = STMEM1D(mask_ratio=args.mask_ratio).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_params += sum(p.numel() for p in [model.patch_embed.parameters(), 
                                                model.encoder_norm.parameters()])
    decoder_params = total_params - encoder_params
    
    print(f'[MODEL] Created STMEM1D:')
    print(f'[MODEL]   Total params: {total_params:,}')
    print(f'[MODEL]   Encoder params: {encoder_params:,} (saved for fine-tuning)')
    print(f'[MODEL]   Decoder params: {decoder_params:,} (discarded)\n')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05
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
    ckpt_mgr = BatchCheckpointManager(args.checkpoint_dir, auto_save_on_interrupt=True)
    start_epoch, start_batch, best_loss = ckpt_mgr.load(model, optimizer, scheduler, device)
    
    # Training loop
    print(f'[TRAINING] Starting from epoch {start_epoch+1}/{args.epochs}, batch {start_batch}/{total_batches}...\n')
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, signals in enumerate(pbar):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                pbar.set_postfix({'status': f'Skipping to batch {start_batch}...'})
                continue
            
            signals = signals.to(device)
            
            # Forward
            loss, _, _ = model(signals)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            
            # Save checkpoint every N batches
            if (batch_idx + 1) % args.checkpoint_every_batches == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                
                ckpt_mgr.save(epoch, batch_idx, total_batches, model, optimizer, scheduler,
                            avg_loss, best_loss, is_best=is_best, epoch_complete=False)
            
            # Check interruption
            if ckpt_mgr.interrupted:
                avg_loss = epoch_loss / (batch_idx + 1)
                ckpt_mgr.save(epoch, batch_idx, total_batches, model, optimizer, scheduler,
                            avg_loss, best_loss, is_best=False, epoch_complete=False)
                print('[TRAINING] Interrupted. Checkpoint saved. Exiting...')
                return
        
        # Reset start_batch for next epoch
        start_batch = 0
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(loader)
        
        # Update scheduler
        scheduler.step()
        
        # Calculate ETA
        total_time = time.time() - training_start_time
        epochs_done = epoch - start_epoch + 1
        avg_time_per_epoch = total_time / epochs_done
        epochs_remaining = args.epochs - (epoch + 1)
        eta_seconds = avg_time_per_epoch * epochs_remaining
        eta_hours = eta_seconds / 3600
        
        print(f'[EPOCH {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}, '
              f'Time: {epoch_time/60:.1f}min, ETA: {eta_hours:.1f}h')
        
        # Save end-of-epoch checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        ckpt_mgr.save(epoch, total_batches - 1, total_batches, model, optimizer, scheduler,
                    avg_loss, best_loss, is_best=is_best, epoch_complete=True)
        
        # Save milestone checkpoints
        if (epoch + 1) % args.save_every == 0:
            milestone_path = args.checkpoint_dir / f'stmem_1d_pretrained_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': {
                    'patch_embed': model.patch_embed.state_dict(),
                    'cls_token': model.cls_token,
                    'pos_embed': model.pos_embed,
                    'sep_tokens': model.sep_tokens,
                    'encoder': model.encoder.state_dict(),
                    'encoder_norm': model.encoder_norm.state_dict(),
                },
                'loss': avg_loss,
            }, milestone_path)
            print(f'[MILESTONE] Saved epoch {epoch+1} checkpoint')
    
    total_time = time.time() - training_start_time
    print(f'\n[DONE] Training completed in {total_time/3600:.1f} hours')
    print(f'[DONE] Best loss: {best_loss:.4f}')
    print(f'[DONE] Final model saved to: {ckpt_mgr.best_model_path}')

# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/processed/1d_signals_100hz')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data (0.01=1%, 1.0=100%)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--mask-ratio', type=float, default=0.75)
    parser.add_argument('--save-every', type=int, default=5, help='Save milestone every N epochs')
    parser.add_argument('--checkpoint-every-batches', type=int, default=500, 
                       help='Save checkpoint every N batches (default: 500 = ~15-20 min)')
    
    args = parser.parse_args()
    train_stmem(args)