"""
ECG → 2D structured image conversion.

Implements the physiologically grounded image representation inspired by
clinical lead placement:

    • RA, LA, LL reference contours
    • Stacked multi-lead layout (H = number of leads)
    • Amplitude clipping to [-3, +3]
    • Linear mapping → [0, 255]
    • Temporal interpolation to fixed width (e.g., 2048)

Output images follow the format:
        (3, H, W)

This representation enables Vision Transformers (ViT) to process ECGs
as structured spatial data while retaining inter-lead dependencies.
"""



from typing import Tuple

import numpy as np
from .resample import pad_or_trim, resample_ecg


def normalize_to_range(signal: np.ndarray, clip_min: float = -3.0, clip_max: float = 3.0) -> np.ndarray:
    """
    Clip and normalize ECG to [0, 255].

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    clip_min : float
        Lower clipping boundary.
    clip_max : float
        Upper clipping boundary.

    Returns
    -------
    np.ndarray
        Float32 array scaled to [0, 255].
    """
    clipped = np.clip(signal, clip_min, clip_max)
    # Map [-3,3] -> [0,255]
    scaled = (clipped - clip_min) / (clip_max - clip_min) * 255.0
    return scaled.astype(np.float32)


def ecg_to_stacked_image(
    signal: np.ndarray,
    fs: float,
    target_fs: float = 400.0,
    target_duration_sec: float = 10.0,
    target_width: int = 2048,
) -> np.ndarray:
    """
    Convert (T, 12) ECG to a 3 × H × W image by stacking leads vertically.

    Steps:
    - Resample to target_fs (if needed)
    - Pad/trim to target_duration_sec
    - Normalize to [0,255]
    - Resize time axis to target_width via linear interpolation
    - Stack leads as rows (H = num_leads)

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, 12) or (T, n_leads).
    fs : float
        Original sampling rate.
    target_fs : float
        Target sampling rate.
    target_duration_sec : float
        Desired duration in seconds.
    target_width : int
        Number of time "pixels" (columns) in output image.

    Returns
    -------
    np.ndarray
        Image array of shape (3, H, W).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    # 1) Resample if necessary
    sig_rs, fs_rs = resample_ecg(signal, fs_in=fs, fs_out=target_fs)

    # 2) Pad or trim to fixed duration
    target_length = int(round(target_duration_sec * fs_rs))
    sig_fixed = pad_or_trim(sig_rs, target_length=target_length)

    # 3) Normalize to [0,255]
    sig_norm = normalize_to_range(sig_fixed)  # shape (T, leads)

    # 4) Resize along time axis to target_width (simple linear interpolation)
    T, n_leads = sig_norm.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_width)
    resized = np.zeros((n_leads, target_width), dtype=np.float32)
    for lead_idx in range(n_leads):
        resized[lead_idx] = np.interp(x_new, x_old, sig_norm[:, lead_idx])

    # resized shape: (leads, W) → want (3, H, W)
    img_single_channel = resized  # (H, W)
    img_rgb = np.stack([img_single_channel] * 3, axis=0)  # (3, H, W)

    return img_rgb


def ecg_batch_to_images(
    signals: np.ndarray,
    fs: float,
    target_fs: float = 400.0,
    target_duration_sec: float = 10.0,
    target_width: int = 2048,
) -> np.ndarray:
    """
    Apply ecg_to_stacked_image to a batch of ECGs.

    Parameters
    ----------
    signals : np.ndarray
        ECG batch of shape (N, T, leads).
    fs : float
        Original sampling rate for all signals.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3, H, W).
    """
    if signals.ndim != 3:
        raise ValueError(f"signals must be 3D (N, T, leads), got shape {signals.shape}")

    images = []
    for i in range(signals.shape[0]):
        img = ecg_to_stacked_image(
            signals[i],
            fs=fs,
            target_fs=target_fs,
            target_duration_sec=target_duration_sec,
            target_width=target_width,
        )
        images.append(img)

    return np.stack(images, axis=0)
