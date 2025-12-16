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

from typing import Tuple, Optional
import warnings

import numpy as np


# -------------------------------------------------------------------------
# Per-lead z-score normalization
# -------------------------------------------------------------------------
def zscore_per_lead(
    signal: np.ndarray, 
    eps: float = 1e-8,
    skip_constant_leads: bool = True
) -> np.ndarray:
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
    skip_constant_leads : bool
        If True, skip normalization for leads with near-zero std.
        Returns original values for those leads with a warning.

    Returns
    -------
    np.ndarray
        Z-score normalized ECG array, same shape as input.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")
    
    if signal.size == 0:
        return signal
    
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    
    # Check for constant leads
    constant_mask = std < eps
    if np.any(constant_mask):
        if skip_constant_leads:
            warnings.warn(
                f"{constant_mask.sum()} leads have near-zero std (<{eps}). "
                "Skipping normalization for these leads."
            )
            # Only normalize non-constant leads
            std_adj = std.copy()
            std_adj[constant_mask] = 1.0  # Avoid division by near-zero
            normalized = (signal - mean) / (std_adj + eps)
            # Restore original values for constant leads
            normalized[:, constant_mask.flatten()] = signal[:, constant_mask.flatten()] - mean[:, constant_mask.flatten()]
            return normalized
        else:
            warnings.warn(
                f"{constant_mask.sum()} leads have near-zero std (<{eps}). "
                "These will be set to zero after normalization."
            )
    
    return (signal - mean) / (std + eps)


# -------------------------------------------------------------------------
# Optional: batch z-score normalization
# -------------------------------------------------------------------------
def zscore_batch(
    batch: np.ndarray,
    eps: float = 1e-8,
    skip_constant_leads: bool = True
) -> np.ndarray:
    """
    Apply per-lead z-score normalization to a batch of ECGs.

    Parameters
    ----------
    batch : np.ndarray
        Array of shape (N, T, leads)
    eps : float
        Small constant to prevent division by zero.
    skip_constant_leads : bool
        If True, skip normalization for leads with near-zero std.

    Returns
    -------
    np.ndarray
        Normalized batch of shape (N, T, leads)
    """
    if batch.ndim != 3:
        raise ValueError(
            f"batch must be 3D (N, T, leads), got shape {batch.shape}"
        )
    
    if batch.size == 0:
        return batch
    
    normalized = []
    for i in range(batch.shape[0]):
        normalized.append(
            zscore_per_lead(batch[i], eps=eps, skip_constant_leads=skip_constant_leads)
        )
    
    return np.stack(normalized, axis=0)


# -------------------------------------------------------------------------
# Dataset-specific normalization
# -------------------------------------------------------------------------
def normalize_dataset(
    signal: np.ndarray,
    dataset: Optional[str] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Apply dataset-specific normalization rules.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    dataset : str, optional
        Dataset identifier. Currently all datasets use z-score.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized ECG array.
    """
    # Currently all datasets use the same z-score normalization
    # This function provides a hook for dataset-specific rules if needed
    return zscore_per_lead(signal, eps=eps)