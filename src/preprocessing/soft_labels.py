# soft_labels.py
# Soft-label utilities (mainly for CODE-15).

from __future__ import annotations

import numpy as np


def hard_to_soft_label(hard: int | float, pos_soft: float = 0.8, neg_soft: float = 0.2) -> float:
    """
    Map hard label {0,1} -> soft probability {neg_soft, pos_soft}.
    """
    h = int(round(float(hard)))
    if h not in (0, 1):
        raise ValueError(f"hard label must be 0/1, got {hard}")
    return float(pos_soft if h == 1 else neg_soft)


def vector_hard_to_soft(hard_labels: np.ndarray, pos_soft: float = 0.8, neg_soft: float = 0.2) -> np.ndarray:
    hard_labels = np.asarray(hard_labels).astype(np.int32, copy=False)
    if not np.all((hard_labels == 0) | (hard_labels == 1)):
        bad = hard_labels[~((hard_labels == 0) | (hard_labels == 1))][:10]
        raise ValueError(f"Labels must be binary 0/1. Examples of bad values: {bad}")
    return np.where(hard_labels == 1, pos_soft, neg_soft).astype(np.float32)
