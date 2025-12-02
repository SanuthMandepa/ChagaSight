"""
ECG Normalization Utilities.

This module provides standardized normalization functions for 12-lead ECG
signals, ensuring numerical stability and compatibility across datasets
with different ranges, devices, and preprocessing histories.

Normalization is a critical step for:
    • Vision Transformer (ViT) image conversion
    • 1D foundation model (ECG-FM) pretraining
    • Hybrid FM–ViT alignment
    • Cross-dataset training (PTB-XL, CODE-15%, SaMi-Trop)

All functions accept and return arrays of shape:
        (num_samples, num_leads)

where num_leads = 12 for standard clinical ECGs.

The primary method implemented here is **per-lead z-score normalization**,
a widely accepted technique in ECG deep learning research.
"""

from typing import Tuple
import numpy as np


# -------------------------------------------------------------------------
# Per-lead z-score normalization
# -------------------------------------------------------------------------
def zscore_per_lead(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-lead z-score normalization.

    For each ECG lead (column), this function computes:
        x_norm = (x - mean) / (std + eps)

    This ensures:
        • Each lead has zero mean and unit variance
        • Improved numerical stability during deep learning
        • Consistency across heterogeneous ECG datasets

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray
        Z-score normalized ECG array, same shape as input.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    return (signal - mean) / (std + eps)


# -------------------------------------------------------------------------
# Optional: batch z-score normalization
# -------------------------------------------------------------------------
def zscore_batch(
    batch: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Apply per-lead z-score normalization to a batch of ECGs.

    Parameters
    ----------
    batch : np.ndarray
        Array of shape (N, T, leads)
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray
        Normalized batch of shape (N, T, leads)
    """
    if batch.ndim != 3:
        raise ValueError(
            f"batch must be 3D (N, T, leads), got shape {batch.shape}"
        )

    return np.stack(
        [zscore_per_lead(ecg, eps=eps) for ecg in batch],
        axis=0
    )
