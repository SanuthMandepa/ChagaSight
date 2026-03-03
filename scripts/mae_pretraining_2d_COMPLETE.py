"""
MAE 2D Pretraining - Professional Production Version

Features:
- Dataset volume control (--subset for testing)
- Robust checkpointing (atomic writes, survives interruptions)
- Auto-resume from checkpoint with correct epoch handling
- Signal handler (saves on Ctrl+C)
- Time tracking and ETA estimation
- Progress tracking with tqdm

Usage:
# Test with 1% data first
python mae_pretraining_2d_COMPLETE.py --subset 0.01 --epochs 5

# Full training
python mae_pretraining_2d_COMPLETE.py --epochs 100
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
# 1. MAE DATASET
# =============================================================================

class MAEImageDataset(Dataset):
    """Dataset for MAE pretraining (all 2D images, no labels needed)."""
    
    def __init__(self, data_dir, subset=1.0, seed=42):
        """
        Args:
            data_dir: Path to data/processed/2d_images
            subset: Fraction of data to use (0.01 = 1%, 1.0 = 100%)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        
        # Find all .npy image files from all datasets
        self.image_paths = []
        for dataset in ['ptbxl', 'samitrop', 'code15']:
            dataset_dir = self.data_dir / dataset
            if dataset_dir.exists():
                paths = sorted(dataset_dir.glob('*.npy'))
                self.image_paths.extend(paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f'No images found in {data_dir}')
        
        # Apply subset for testing
        if subset < 1.0:
            np.random.seed(seed)
            n_samples = int(len(self.image_paths) * subset)
            indices = np.random.choice(len(self.image_paths), n_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(indices)]
        
        print(f'[DATA] Loaded {len(self.image_paths)} images ({subset*100:.1f}% of data)')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image: (3, 24, 2048) uint8
        img = np.load(self.image_paths[idx])
        img = torch.from_numpy(img).float() / 255.0  # [0, 1]
        return img

# =============================================================================
# 2. MAE MODEL ARCHITECTURE
# =============================================================================

class PatchEmbed2D(nn.Module):
    """2D Patch Embedding for ECG images."""
    def __init__(self, img_size=(24, 2048), patch_size=(8, 64), in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 96
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, 768, 3, 32)
        x = x.flatten(2).transpose(1, 2)  # (B, 96, 768)
        return x

class MAE2D(nn.Module):
    """
    Masked Autoencoder for 2D ECG images.
    
    Paper: He et al. (2022) + Kim et al. (2025)
    Architecture:
    - Encoder: 12-layer ViT (saved for fine-tuning)
    - Decoder: 4-layer lightweight (discarded after pretraining)
    - Mask ratio: 75%
    - Loss: MSE on masked patches only
    """
    
    def __init__(self, img_size=(24, 2048), patch_size=(8, 64), 
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        
        self.patch_embed = PatchEmbed2D(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Encoder (saved for fine-tuning)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
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
        
        # Decoder (discarded after pretraining)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * 3)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        """Random masking of patches."""
        B, N, D = x.shape  # (B, 96, 768)
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
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
    
    def forward_loss(self, imgs, pred, mask):
        """MSE loss on masked patches only."""
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def patchify(self, imgs):
        """Convert images to patches."""
        B, C, H, W = imgs.shape
        p_h, p_w = self.patch_size
        h, w = H // p_h, W // p_w
        
        x = imgs.reshape(B, C, h, p_h, w, p_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p_h * p_w * C)
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask

# =============================================================================
# 3. CHECKPOINT MANAGER (FIXED)
# =============================================================================

class CheckpointManager:
    """Handles checkpointing with atomic writes and auto-resume."""
    
    def __init__(self, checkpoint_dir, auto_save_on_interrupt=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.checkpoint_dir / 'mae_2d_checkpoint.pt'
        self.best_model_path = self.checkpoint_dir / 'mae_2d_pretrained.pt'
        
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
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)


        
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

def train_mae(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*70}')
    print(f'MAE 2D Pretraining')
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
    dataset = MAEImageDataset(args.data_dir, subset=args.subset)
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
    model = MAE2D(mask_ratio=args.mask_ratio).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f'[MODEL] Created MAE2D:')
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
    
    # Scheduler with warmup
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
        for imgs in pbar:
            imgs = imgs.to(device)
            
            # Forward
            loss, _, _ = model(imgs)
            
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
    parser = argparse.ArgumentParser(description='MAE 2D Pretraining')
    
    # Data
    parser.add_argument('--data-dir', type=str, 
                       default='data/processed/2d_images',
                       help='Path to 2D images directory')
    parser.add_argument('--subset', type=float, default=1.0,
                       help='Dataset subset (0.01=1%%, 1.0=100%%). Use 0.01 for testing!')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
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
    
    train_mae(args)