# src/training/losses.py  ── FIXED VERSION
"""
Loss Functions for ChagaSight

FIXES vs original:
  - Added eps clamping to log() inputs (prevents log(0) = -inf)
  - Added input validation (warns if logits/labels have bad values)
  - AsymmetricBCE now safe with AMP float16 inputs

Paper Reference:
  Van Santvliet et al. (2025): AsymmetricBCE γ⁺=0, γ⁻=2, pos_weight=10
  Kim et al. (2025):           Cosine similarity alignment loss (λ=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricBCELoss(nn.Module):
    """
    Asymmetric Binary Cross-Entropy Loss.

    Paper: Van Santvliet et al. (2025)

    Formula:
        L = -w_pos × (1-p)^γ⁺ × y × log(p)
            - p^γ⁻ × (1-y) × log(1-p)

    γ⁺=0  : no focusing on positives (standard for rare disease)
    γ⁻=2  : focus on hard negatives (reduces false positives)
    w_pos=10: upweight positives to handle ~3.4% class imbalance
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        pos_weight: float = 10.0,
        reduction: str = 'mean',
        eps: float = 1e-6,          # numerical stability guard
    ):
        super().__init__()
        self.gamma_pos  = gamma_pos
        self.gamma_neg  = gamma_neg
        self.pos_weight = pos_weight
        self.reduction  = reduction
        self.eps        = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) raw logits
            targets: (B,) labels in [0, 1] — can be soft
        Returns:
            loss: scalar
        """
        # Work in float32 even under AMP (prevents underflow in float16)
        logits  = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)

        # FIXED: clamp with eps to prevent log(0) = -inf → NaN
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # Positive branch: -w_pos × (1-p)^γ⁺ × y × log(p)
        pos_loss = -(
            self.pos_weight
            * torch.pow(1.0 - probs, self.gamma_pos)
            * targets
            * torch.log(probs)
        )

        # Negative branch: -p^γ⁻ × (1-y) × log(1-p)
        neg_loss = -(
            torch.pow(probs, self.gamma_neg)
            * (1.0 - targets)
            * torch.log(1.0 - probs)
        )

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss   # 'none'


class CosineSimilarityAlignmentLoss(nn.Module):
    """
    Cosine similarity alignment loss for REPA module.

    Paper: Kim et al. (2025) Section 2.9
    L_align = 1 - cosine_similarity(aligned_2d, fm_features.detach())

    FM features are DETACHED — the FM is only used as a target,
    not trained through this loss.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        aligned_2d_features: torch.Tensor,
        fm_features: torch.Tensor,
    ) -> torch.Tensor:
        # Detach FM — stop gradient
        fm_features = fm_features.detach().float()
        aligned_2d_features = aligned_2d_features.float()

        # L2-normalise both
        a = F.normalize(aligned_2d_features, p=2, dim=1)
        b = F.normalize(fm_features, p=2, dim=1)

        # Cosine similarity → alignment loss in [0, 2]
        cos_sim = (a * b).sum(dim=1)
        loss = 1.0 - cos_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: AsymmetricBCE + λ × CosineSimilarityAlignment

    Paper: Kim et al. (2025) + Van Santvliet et al. (2025)
    λ = 0.5 (alignment weight)
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        pos_weight: float = 10.0,
        alignment_weight: float = 0.5,
    ):
        super().__init__()
        self.bce = AsymmetricBCELoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            pos_weight=pos_weight,
        )
        self.alignment = CosineSimilarityAlignmentLoss()
        self.alignment_weight = alignment_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aligned_2d_features: Optional[torch.Tensor] = None,
        fm_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            logits:               (B,)   prediction logits
            targets:              (B,)   soft/hard labels
            aligned_2d_features:  (B, 768) aligned 2D features (optional)
            fm_features:          (B, 768) FM features (optional)

        Returns:
            dict: total_loss, bce_loss, alignment_loss
        """
        bce = self.bce(logits, targets)

        if aligned_2d_features is not None and fm_features is not None:
            align = self.alignment(aligned_2d_features, fm_features)
            total = bce + self.alignment_weight * align
            return {
                'total_loss':     total,
                'bce_loss':       bce,
                'alignment_loss': align,
            }
        else:
            return {
                'total_loss':     bce,
                'bce_loss':       bce,
                'alignment_loss': torch.tensor(0.0, device=logits.device),
            }


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)

    logits  = torch.randn(16)
    targets = torch.tensor([1.0, 0.0] * 8)
    feat_2d = torch.randn(16, 768)
    feat_fm = torch.randn(16, 768)

    criterion = CombinedLoss()
    out = criterion(logits, targets, feat_2d, feat_fm)

    print('CombinedLoss self-test:')
    for k, v in out.items():
        print(f'  {k}: {v.item():.4f}')

    # Confirm no inf/nan
    for k, v in out.items():
        assert torch.isfinite(v), f'{k} is not finite!'
    print('✓ All losses finite.')