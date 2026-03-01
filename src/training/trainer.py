# src/training/trainer.py  ── v6 FINAL
"""
Trainer with Progressive Unfreezing — Definitive Version

═══════════════════════════════════════════════════════════════════
ISSUES FOUND AND FIXED IN v6
═══════════════════════════════════════════════════════════════════

CRITICAL BUG #A — "Val subset had only one class" (every validation)
  Root cause: val_loader has shuffle=False. Val CSV is ordered by dataset
  (PTB-XL first ~4367 rows, then SaMi-Trop, then CODE-15). When taking
  first 2000 samples, ALL come from PTB-XL which has 0 Chagas cases.
  The one-class guard triggers and falls back to FULL validation every time.
  
  Impact: Every mid-training validation check takes 8-10 min instead of ~2 min.
  Phase 2 has 24 validation checks → +3.2 hours wasted.
  
  Fix: _build_val_subset() pre-computes a STRATIFIED random subset of val indices
  at trainer init time, guaranteeing both classes are always present.
  Uses numpy to sample separately from positive/negative indices.

CRITICAL BUG #B — Phase 2 uses grad_accum=4 (same as Phase 1)
  Root cause: v3 trainer has single grad_accum_steps for both phases.
  Phase 1: 85M params frozen → accum=4 matches paper eff.batch=64, fast.
  Phase 2: ALL 173M params training → accum=4 = ~30 hours on RTX 3050.
  
  Fix: Separate phase1_grad_accum and phase2_grad_accum parameters.
  phase1_grad_accum=4  → eff.batch=64, ~35 min ✓
  phase2_grad_accum=1  → eff.batch=16, ~8 hours ✓

BUG #C — metrics.py compute_metrics() called without fast=True
  Root cause: v3 trainer always uses 10000 permutations.
  Mid-training validation only needs a quick estimate.
  
  Fix: fast=True → 1000 permutations (~0.1s vs ~1.5s per call).
  End-of-phase always uses 10000 permutations (official).

BACKWARD COMPATIBILITY:
  Old param grad_accum_steps still accepted (sets both phase accums equally).
  trainer.py is drop-in replaceable — notebook Cell 5 needs one update.

═══════════════════════════════════════════════════════════════════
ALL PREVIOUS FIXES STILL PRESENT
═══════════════════════════════════════════════════════════════════
FIX #1: Gradient clipping (max_grad_norm=1.0) → prevented loss=inf
FIX #2: torch.amp API → was using deprecated torch.cuda.amp
FIX #3: nan/inf loss guard → skips bad batches
FIX #4: LR warmup (200 steps) → stabilises random-init training
FIX #5: Gradient accumulation → matches paper's effective batch
FIX #6: Smoothed loss display (50-iter rolling average)
FIX #7: "Validating..." description → no frozen appearance
FIX #8: Inner tqdm bar in validate() → progress visible
FIX #9: val_subset_size → fast mid-training validation
FIX #10: lr_scheduler UserWarning suppressed
FIX #11: Phase-specific grad_accum (this file)

Paper: Van Santvliet et al. (2025)
  Phase 1: FM frozen, 2000 iters, LR=2e-4, accum=4
  Phase 2: FM unfrozen, differential LR, 12000 iters, accum=1
"""

import warnings
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .losses import CombinedLoss
from .metrics import compute_metrics


class ChagasTrainer:
    """
    Definitive progressive-unfreezing trainer for ChagaSight.

    Parameters
    ----------
    phase1_grad_accum : int
        Gradient accumulation for Phase 1 (FM frozen, cheap).
        Default 4 → effective batch = 16×4 = 64 (matches paper).
    phase2_grad_accum : int
        Gradient accumulation for Phase 2 (all 173M params training).
        Default 1 → Phase 2 takes ~8h not ~30h on RTX 3050 6GB.
    grad_accum_steps : int
        Legacy param. If provided, sets both phase accums equally.
        (kept for backward compatibility with existing notebooks)
    val_subset_size : int or None
        Mid-training validation uses a STRATIFIED subset of this size.
        Guarantees both classes present (fixes the one-class bug).
        Full val set always used at end of each phase.
        Default 3000 is large enough to always include both classes.
    val_n_permutations : int
        Permutations for mid-training TPR@5% calculation.
        Default 1000 (fast, ~0.1s). End-of-phase always uses 10000.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        phase1_iterations: int = 2000,
        phase2_iterations: int = 12000,
        phase1_lr: float = 2e-4,
        phase2_lr_high: float = 2e-4,
        phase2_lr_low: float = 2e-5,
        checkpoint_dir: str = 'checkpoints',
        use_amp: bool = True,
        val_every_n_iters: int = 500,
        max_grad_norm: float = 1.0,
        warmup_iters: int = 200,
        # Phase-specific gradient accumulation (FIX #B)
        phase1_grad_accum: int = 4,
        phase2_grad_accum: int = 1,
        # Legacy: single accum for both phases (overrides if provided explicitly)
        grad_accum_steps: Optional[int] = None,
        # Val subset (FIX #A)
        val_subset_size: Optional[int] = 3000,
        val_n_permutations: int = 1000,   # FIX #C: fast mid-training metric
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.phase1_iterations  = phase1_iterations
        self.phase2_iterations  = phase2_iterations
        self.phase1_lr          = phase1_lr
        self.phase2_lr_high     = phase2_lr_high
        self.phase2_lr_low      = phase2_lr_low
        self.val_every_n_iters  = val_every_n_iters
        self.max_grad_norm      = max_grad_norm
        self.warmup_iters       = warmup_iters
        self.val_subset_size    = val_subset_size
        self.val_n_permutations = val_n_permutations

        # Handle backward-compatible grad_accum_steps
        if grad_accum_steps is not None:
            self.phase1_grad_accum = grad_accum_steps
            self.phase2_grad_accum = grad_accum_steps
        else:
            self.phase1_grad_accum = phase1_grad_accum
            self.phase2_grad_accum = phase2_grad_accum

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = use_amp
        self.scaler  = GradScaler('cuda') if use_amp else None

        self.criterion = CombinedLoss()

        self.best_val_score    = 0.0
        self.current_phase     = 1
        self.current_iteration = 0
        self.skipped_batches   = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_auroc':  [], 'val_tpr_5pct': [],
            'grad_norm':  [],
        }

        # Pre-compute stratified val subset (FIX #A)
        self._val_subset_indices = None
        self._val_subset_loader  = None
        if val_subset_size is not None:
            self._build_val_subset()

    # ── FIX #A: Stratified val subset ────────────────────────────────

    def _build_val_subset(self):
        """
        Pre-compute a stratified random subset of the validation set.

        Samples separately from positive and negative examples so that
        BOTH CLASSES are always present regardless of dataset ordering.
        This permanently fixes the 'only one class' fallback.

        Called once at trainer init — no runtime overhead during training.
        """
        dataset = self.val_loader.dataset

        # Collect all labels from the val dataset
        if hasattr(dataset, 'df'):
            # Our ChagasDataset stores a DataFrame with label_hard column
            labels = dataset.df['label_hard'].values.astype(int)
        else:
            # Fallback: iterate (slow but only done once)
            print("  Building val subset: scanning labels (one-time, ~30s)...")
            labels = []
            for i in range(len(dataset)):
                item = dataset[i]
                labels.append(int(item['hard_label'].item()))
            labels = np.array(labels)

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        n_pos_total = len(pos_idx)
        n_neg_total = len(neg_idx)

        # Ensure at least 30 positives for stable TPR@5% estimate
        n_pos = min(max(30, self.val_subset_size // 10), n_pos_total)
        n_neg = min(self.val_subset_size - n_pos, n_neg_total)

        rng = np.random.default_rng(42)
        chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        chosen = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(chosen)  # mix pos/neg randomly

        self._val_subset_indices = chosen.tolist()

        # Build a DataLoader for the subset
        subset_dataset = Subset(dataset, self._val_subset_indices)
        batch_size = getattr(self.val_loader, 'batch_size', 16) or 16
        self._val_subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(self.val_loader, 'num_workers', 0),
            pin_memory=getattr(self.val_loader, 'pin_memory', False),
            collate_fn=self.val_loader.collate_fn
                       if hasattr(self.val_loader, 'collate_fn') else None,
        )

        print(f"  ✓ Val subset: {n_pos} positives + {n_neg} negatives "
              f"= {n_pos+n_neg} samples (stratified, guaranteed both classes)")

    # ── Freeze / Unfreeze ─────────────────────────────────────────────

    def freeze_fm(self):
        for p in self.model.vit_1d_fm.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.model.vit_1d_fm.parameters())
        print(f"✓ FM frozen  ({n:,} params frozen)")

    def unfreeze_fm(self):
        for p in self.model.vit_1d_fm.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in self.model.vit_1d_fm.parameters())
        print(f"✓ FM unfrozen  ({n:,} params now trainable)")

    # ── Optimisers ────────────────────────────────────────────────────

    def get_optimizer(self, phase: int) -> AdamW:
        if phase == 1:
            return AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.phase1_lr, weight_decay=1e-4, betas=(0.9, 0.999),
            )
        return AdamW([
            {'params': self.model.vit_2d.parameters(),    'lr': self.phase2_lr_low},
            {'params': self.model.vit_1d_fm.parameters(), 'lr': self.phase2_lr_low},
            {'params': self.model.repa.parameters(),       'lr': self.phase2_lr_high},
            {'params': self.model.classifier.parameters(), 'lr': self.phase2_lr_high},
        ], weight_decay=1e-4, betas=(0.9, 0.999))

    # ── LR warmup ─────────────────────────────────────────────────────

    def get_warmup_scheduler(self, optimizer: AdamW, warmup_iters: int):
        def _lr_lambda(step: int) -> float:
            return float(step) / float(max(1, warmup_iters)) if step < warmup_iters else 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Single training step ──────────────────────────────────────────

    def train_iteration(self, optimizer, train_iter, scheduler=None,
                        accum_step: int = 0, grad_accum: int = 1):
        self.model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(self.train_loader)
            batch = next(train_iter)

        images  = batch['image'].to(self.device, non_blocking=True)
        signals = batch['signal'].to(self.device, non_blocking=True)
        ages    = batch['age'].to(self.device, non_blocking=True)
        sexes   = batch['sex'].to(self.device, non_blocking=True)
        labels  = batch['label'].to(self.device, non_blocking=True)

        is_last_accum = (accum_step + 1) % grad_accum == 0

        with autocast('cuda', enabled=self.use_amp):
            outputs = self.model(images, signals, ages, sexes)
            losses  = self.criterion(
                outputs['logits'].squeeze(), labels,
                outputs['aligned_2d_features'], outputs['fm_features'],
            )
            loss = losses['total_loss'] / grad_accum

        if not torch.isfinite(loss):
            self.skipped_batches += 1
            if self.skipped_batches <= 5 or self.skipped_batches % 20 == 0:
                print(f"\n⚠️  Non-finite loss (total skips: {self.skipped_batches})")
            optimizer.zero_grad()
            return float('nan'), train_iter

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if is_last_accum:
                self.scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if scheduler: scheduler.step()
                self.history['grad_norm'].append(float(gn))
        else:
            loss.backward()
            if is_last_accum:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if scheduler: scheduler.step()
                self.history['grad_norm'].append(float(gn))

        return (loss * grad_accum).item(), train_iter

    # ── Validation ────────────────────────────────────────────────────

    def _run_val_loop(self, loader, max_batches: int, desc: str) -> tuple:
        """Run one validation loop and return (avg_loss, all_probs, all_labels)."""
        all_probs, all_labels, total_loss = [], [], 0.0

        val_pbar = tqdm(loader, total=max_batches, desc=desc,
                        leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                if batch_idx >= max_batches:
                    break
                images      = batch['image'].to(self.device, non_blocking=True)
                signals     = batch['signal'].to(self.device, non_blocking=True)
                ages        = batch['age'].to(self.device, non_blocking=True)
                sexes       = batch['sex'].to(self.device, non_blocking=True)
                labels      = batch['label'].to(self.device, non_blocking=True)
                hard_labels = batch['hard_label']

                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images, signals, ages, sexes)
                    losses  = self.criterion(
                        outputs['logits'].squeeze(), labels,
                        outputs['aligned_2d_features'], outputs['fm_features'],
                    )

                all_probs.append(torch.sigmoid(outputs['logits']).cpu().numpy())
                all_labels.append(hard_labels.numpy())
                total_loss += losses['total_loss'].item()

        val_pbar.close()
        avg_loss = total_loss / max(1, batch_idx + 1)
        return avg_loss, np.concatenate(all_probs), np.concatenate(all_labels)

    def validate(self, fast: bool = False) -> tuple:
        """
        Validate on the validation set.

        fast=True: Uses pre-computed STRATIFIED subset (FIX #A) + 1000 perms (FIX #C).
                   No "only one class" fallback needed — subset always has both classes.
                   Time: ~30-60s on RTX 3050.
        fast=False: Uses full val set + 10000 perms (official, end-of-phase only).
                    Time: ~5-8 min.

        Returns (avg_loss, metrics_dict).
        """
        self.model.eval()

        if fast and self._val_subset_loader is not None:
            # Use pre-computed stratified subset (GUARANTEED to have both classes)
            n_total = len(self._val_subset_indices)
            n_pos   = int(sum(1 for i in self._val_subset_indices
                             if self.val_loader.dataset.df['label_hard'].iloc[i] == 1)
                         if hasattr(self.val_loader.dataset, 'df') else 0)
            desc = f"Val subset ({n_total} samples, stratified)"
            avg_loss, all_probs, all_labels = self._run_val_loop(
                self._val_subset_loader,
                max_batches=len(self._val_subset_loader),
                desc=desc,
            )
            # FIX #C: Use fewer permutations for fast mid-training checks
            metrics = compute_metrics(all_labels, all_probs,
                                      num_permutations=self.val_n_permutations)
        else:
            # Full validation (end of phase)
            desc = f"Val full ({len(self.val_loader)} batches)"
            avg_loss, all_probs, all_labels = self._run_val_loop(
                self.val_loader,
                max_batches=len(self.val_loader),
                desc=desc,
            )
            # Full permutations for official score
            metrics = compute_metrics(all_labels, all_probs, num_permutations=10000)

        return avg_loss, metrics

    # ── Checkpoint helpers ────────────────────────────────────────────

    def save_checkpoint(self, fold, phase, iteration, val_score, metrics, optimizer,
                        is_best=False):
        state = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'phase': phase, 'iteration': iteration,
            'val_score': val_score, 'best_val_score': self.best_val_score,
            'metrics': metrics, 'history': self.history,
            'skipped_batches': self.skipped_batches,
        }
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(state, self.checkpoint_dir / f'fold{fold}_latest.pt')
        if is_best:
            torch.save(state, self.checkpoint_dir / f'fold{fold}_best.pt')
            print(f"\n  ✅ NEW BEST → TPR@5%={val_score:.4f}")

    def load_checkpoint(self, path, optimizer=None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_phase     = ckpt.get('phase', 1)
        self.current_iteration = ckpt.get('iteration', 0)
        self.best_val_score    = ckpt.get('best_val_score', 0.0)
        self.history           = ckpt.get('history', self.history)
        self.skipped_batches   = ckpt.get('skipped_batches', 0)
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"✓ Resumed  Phase={self.current_phase}  Iter={self.current_iteration}"
              f"  Best={self.best_val_score:.4f}")
        return ckpt

    # ── Shared phase training loop ────────────────────────────────────

    def _run_phase(self, fold, phase, total_iters, start_iter,
                   optimizer, scheduler, grad_accum) -> Dict:
        label      = f"Phase {phase}"
        train_iter = iter(self.train_loader)
        progress   = tqdm(total=total_iters, initial=start_iter,
                          desc=label, dynamic_ncols=True)
        running_loss = []
        accum_step   = 0

        for iteration in range(start_iter, total_iters):
            loss, train_iter = self.train_iteration(
                optimizer, train_iter, scheduler, accum_step, grad_accum
            )
            accum_step = (accum_step + 1) % grad_accum

            if loss == loss:  # not nan
                running_loss.append(loss)
                if len(running_loss) > 50:
                    running_loss.pop(0)

            progress.update(1)
            smooth = np.mean(running_loss) if running_loss else float('nan')

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lr_str = f"{scheduler.get_last_lr()[0]:.2e}"
            except Exception:
                lr_str = "N/A"

            progress.set_postfix(loss=f'{smooth:.4f}', skips=self.skipped_batches, lr=lr_str)
            self.current_iteration = iteration + 1
            self.history['train_loss'].append(smooth)

            # Mid-training validation (fast stratified subset)
            if (iteration + 1) % self.val_every_n_iters == 0:
                progress.set_description(f"{label} | Validating...")
                val_loss, metrics = self.validate(fast=True)
                val_score = metrics.get('tpr_5pct', 0.0)

                self.history['val_tpr_5pct'].append(val_score)
                self.history['val_auroc'].append(metrics.get('auroc', 0.0))
                self.history['val_loss'].append(val_loss)

                is_best = val_score > self.best_val_score
                if is_best:
                    self.best_val_score = val_score
                self.save_checkpoint(fold, phase, iteration + 1,
                                     val_score, metrics, optimizer, is_best)

                n_perms = metrics.get('num_permutations', self.val_n_permutations)
                method = "OFF." if metrics.get('using_official') else "APPROX."
                progress.set_description(
                    f"{label} | TPR@5%={val_score:.4f} "
                    f"AUROC={metrics.get('auroc',0):.4f} [{method}{n_perms}p]"
                )

        progress.close()

        # End-of-phase: FULL validation with 10000 permutations
        print(f"\n  Running FULL validation at end of Phase {phase} (10000 perms)...")
        val_loss, metrics = self.validate(fast=False)
        val_score = metrics.get('tpr_5pct', 0.0)
        method = "OFFICIAL" if metrics.get('using_official') else "APPROX"
        print(f"  ✓ Phase {phase} complete:")
        print(f"    TPR@5%:  {val_score:.4f}  [{method}]")
        print(f"    AUROC:   {metrics.get('auroc',0):.4f}")
        print(f"    AUPRC:   {metrics.get('auprc',0):.4f}")
        print(f"    Samples: {metrics.get('n_total',0)} total, {metrics.get('n_pos',0)} positive")

        is_best = val_score > self.best_val_score
        if is_best:
            self.best_val_score = val_score
        self.save_checkpoint(fold, phase, total_iters, val_score, metrics, optimizer, is_best)
        return metrics

    # ── Main entry point ──────────────────────────────────────────────

    def train(self, fold: int = 0, resume_from: Optional[str] = None) -> Dict:
        """
        Two-phase training with progressive FM unfreezing.

        Phase 1 (FM frozen):   2000 iters, LR=2e-4, accum=4 → ~35 min
        Phase 2 (FM unfrozen): 12000 iters, differential LR, accum=1 → ~8h
        """
        print(f"\n{'='*68}")
        print(f"  ChagaSight Training — Fold {fold}")
        print(f"  Phase 1: LR={self.phase1_lr:.0e}  accum={self.phase1_grad_accum}"
              f"  eff.batch={16*self.phase1_grad_accum}")
        print(f"  Phase 2: LR={self.phase2_lr_low:.0e}/{self.phase2_lr_high:.0e}"
              f"  accum={self.phase2_grad_accum}"
              f"  eff.batch={16*self.phase2_grad_accum}")
        print(f"  Grad clip: {self.max_grad_norm}  |  Warmup: {self.warmup_iters}"
              f"  |  Val subset: {self.val_subset_size}  |  Perms(fast): {self.val_n_permutations}")
        print(f"{'='*68}")

        if resume_from and Path(resume_from).exists():
            self.load_checkpoint(resume_from)
            start_phase = self.current_phase
            start_iter  = self.current_iteration
            if start_phase == 2:
                self.unfreeze_fm()
        else:
            start_phase = 1
            start_iter  = 0
            self.current_phase = self.current_iteration = 0

        # ── Phase 1 ─────────────────────────────────────────────────
        if start_phase == 1:
            eta = self.phase1_iterations * (16 * self.phase1_grad_accum / 64) * 1.1 / 60
            print(f"\n📌 PHASE 1: FM Frozen — {self.phase1_iterations} iters"
                  f" | ETA ~{eta:.0f} min")

            if start_iter == 0:
                self.freeze_fm()

            optimizer = self.get_optimizer(phase=1)
            scheduler = self.get_warmup_scheduler(optimizer, self.warmup_iters)

            if resume_from and start_iter > 0:
                ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            self._run_phase(fold, 1, self.phase1_iterations, start_iter,
                            optimizer, scheduler, self.phase1_grad_accum)

            self.current_phase = 2
            self.current_iteration = 0
            start_phase = 2
            start_iter  = 0

        # ── Phase 2 ─────────────────────────────────────────────────
        if start_phase == 2:
            eta_h = self.phase2_iterations * 2.5 / 3600
            print(f"\n📌 PHASE 2: FM Unfrozen — {self.phase2_iterations} iters"
                  f" | ETA ~{eta_h:.1f}h")
            print(f"  LR — FM/2D: {self.phase2_lr_low:.1e}  |  Head: {self.phase2_lr_high:.1e}")

            if start_iter == 0:
                self.unfreeze_fm()

            optimizer = self.get_optimizer(phase=2)
            scheduler = self.get_warmup_scheduler(optimizer, max(1, self.warmup_iters // 2))

            if resume_from and start_iter > 0:
                ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            final_metrics = self._run_phase(
                fold, 2, self.phase2_iterations, start_iter,
                optimizer, scheduler, self.phase2_grad_accum
            )

            print(f"\n{'='*68}")
            print(f"  Training complete — Fold {fold}")
            print(f"  Best TPR@5%:            {self.best_val_score:.4f}")
            print(f"  Final AUROC:            {final_metrics.get('auroc',0):.4f}")
            print(f"  Skipped batches (nan):  {self.skipped_batches}")
            if self.skipped_batches > 50:
                print("  ⚠️  High skip count — check signal normalisation.")
            print(f"{'='*68}")
            return final_metrics

        return {}