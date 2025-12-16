"""
ECG → 2D image embedding utilities (RA/LA/LL contour mapping).

This module converts 12-lead ECG signals into physiologically
structured 2D images, following the idea from:

    "Embedding ECG Signals into 2D Image with Preserved Spatial
     Information for Chagas Disease Classification" (Kim et al., 2025)

Key ideas:
    - We treat the 12-lead ECG as projections of a body-surface
      potential map recorded at nine electrode sites.
    - We build 3 channels (analogous to RGB), each corresponding
      to a contour defined relative to one of the limb electrodes:
        * Channel 1: LL, V1–V6, LA  (reference ≈ RA)
        * Channel 2: RA, V1–V6, LL  (reference ≈ LA)
        * Channel 3: RA, V1–V6, LA  (reference ≈ LL)
    - Each channel is an 8×T map (8 leads along the "spatial" axis).
      We then pad this to 24 rows and resize the time axis to a
      target width (default 2048), yielding a final image of size:

          (C, H, W) = (3, 24, 2048)

Inputs:
    - Assumes signals of shape (T, 12), where T is number of samples
      and the 12 leads are ordered as in PTB-XL WFDB records:

        [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]

Usage:
    - Stage 1 (already handled elsewhere) should perform:
        * baseline removal
        * resampling to a common fs
        * padding/trimming to a fixed duration
        * per-lead z-score normalization
    - This module focuses purely on:
        * amplitude clipping and scaling
        * RA/LA/LL contour construction
        * resizing to (3, 24, target_width)
"""

from typing import Tuple, Union
import warnings

import numpy as np


def _clip_and_scale(
    channel: np.ndarray, 
    clip_min: float = -3.0, 
    clip_max: float = 3.0
) -> np.ndarray:
    """
    Clip and linearly map channel amplitudes to [0, 255].

    Parameters
    ----------
    channel : np.ndarray
        Channel array of any shape.
    clip_min : float
        Lower clipping boundary before scaling.
    clip_max : float
        Upper clipping boundary before scaling.

    Returns
    -------
    np.ndarray
        Float32 array scaled to [0, 255], same shape as input.
    """
    if clip_min >= clip_max:
        raise ValueError(f"clip_min ({clip_min}) must be less than clip_max ({clip_max})")
    
    clipped = np.clip(channel, clip_min, clip_max)
    
    # Check if all values are within clipping range (warning only)
    out_of_bounds = np.logical_or(channel < clip_min, channel > clip_max)
    if np.any(out_of_bounds):
        warnings.warn(
            f"{out_of_bounds.sum()} values clipped ({out_of_bounds.mean()*100:.1f}%) "
            f"to range [{clip_min}, {clip_max}]"
        )
    
    # Map [clip_min, clip_max] → [0, 255]
    scaled = (clipped - clip_min) / (clip_max - clip_min) * 255.0
    return scaled.astype(np.float32)


def _resize_time_axis(rows: np.ndarray, target_width: int) -> np.ndarray:
    """
    Resize along the temporal axis using 1D linear interpolation.

    Parameters
    ----------
    rows : np.ndarray
        Array of shape (H, T) – e.g. 8 or 24 rows × time.
    target_width : int
        Desired number of time "pixels" (columns).

    Returns
    -------
    np.ndarray
        Array of shape (H, target_width).
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2D (H, T), got shape {rows.shape}")
    
    if target_width <= 0:
        raise ValueError(f"target_width must be positive, got {target_width}")

    H, T = rows.shape
    if T == target_width:
        return rows
    
    if T < 2 or target_width < 2:
        raise ValueError(f"Both original T ({T}) and target_width ({target_width}) must be >= 2")

    x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_width, dtype=np.float32)

    out = np.zeros((H, target_width), dtype=np.float32)
    for i in range(H):
        out[i] = np.interp(x_new, x_old, rows[i])

    return out


def ecg_to_contour_image(
    signal: np.ndarray,
    target_width: int = 2048,
    clip_range: Tuple[float, float] = (-3.0, 3.0),
    check_normalization: bool = True
) -> np.ndarray:
    """
    Convert a 12-lead ECG (T, 12) into a contour-based 2D image (3, 24, W).

    Steps:
        1) Construct 3 channels using RA/LA/LL contours with subtraction:
               Channel 1: [LL-RA, V1-RA, V2-RA, V3-RA, V4-RA, V5-RA, V6-RA, LA-RA]
               Channel 2: [RA-LA, V1-LA, V2-LA, V3-LA, V4-LA, V5-LA, V6-LA, LL-LA]
               Channel 3: [RA-LL, V1-LL, V2-LL, V3-LL, V4-LL, V5-LL, V6-LL, LA-LL]
           (Each is an 8×T map.)
        2) Clip each channel to [-3, 3] and scale to [0, 255].
        3) Pad each 8×T map to 24×T along the spatial axis.
        4) Resize the time axis to target_width via linear interpolation.
        5) Stack 3 channels → (3, 24, target_width).

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, 12). Lead order is assumed to be:
          [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
        The signal should be z-scored per lead.
    target_width : int
        Number of time "pixels" in the final image (e.g. 2048).
    clip_range : Tuple[float, float]
        Clipping range for each channel before scaling to [0, 255].
    check_normalization : bool
        If True, warn if signal doesn't appear to be z-score normalized.

    Returns
    -------
    np.ndarray
        Structured ECG image of shape (3, 24, target_width).
    """
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(
            f"signal must have shape (T, 12) with PTB-XL lead ordering, got {signal.shape}"
        )
    
    T = signal.shape[0]
    if T < 100:
        warnings.warn(f"Signal length T={T} is very short, may lose temporal information")
    
    # Check if signal appears to be normalized
    if check_normalization:
        means = signal.mean(axis=0)
        stds = signal.std(axis=0)
        if np.any(np.abs(means) > 1.0) or np.any(np.abs(stds - 1.0) > 0.5):
            warnings.warn(
                "Signal may not be properly z-score normalized. "
                f"Mean range: [{means.min():.2f}, {means.max():.2f}], "
                f"Std range: [{stds.min():.2f}, {stds.max():.2f}]"
            )

    # Transpose to (12, T) for easier lead indexing
    leads = signal.T  # (12, T)

    # According to PTB-XL:
    #   0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6–11: V1–V6
    aVR = leads[3]
    aVL = leads[4]
    aVF = leads[5]
    v1_v6 = leads[6:12]  # shape (6, T)

    # Approximate limb electrodes from augmented leads
    ra = aVR  # approx RA-referenced signal
    la = aVL  # approx LA-referenced signal
    ll = aVF  # approx LL-referenced signal

    # 1) Build 3 contour channels (each 8×T) with subtraction
    # Channel 1: [LL-RA, V1-RA, V2-RA, ..., V6-RA, LA-RA]
    ch1_rows = np.vstack([ll - ra, v1_v6 - ra, la - ra])  # (8, T)

    # Channel 2: [RA-LA, V1-LA, V2-LA, ..., V6-LA, LL-LA]
    ch2_rows = np.vstack([ra - la, v1_v6 - la, ll - la])  # (8, T)

    # Channel 3: [RA-LL, V1-LL, V2-LL, ..., V6-LL, LA-LL]
    ch3_rows = np.vstack([ra - ll, v1_v6 - ll, la - ll])  # (8, T)

    # 2) Clip and scale each channel to [0, 255]
    clip_min, clip_max = clip_range
    ch1_rows = _clip_and_scale(ch1_rows, clip_min=clip_min, clip_max=clip_max)
    ch2_rows = _clip_and_scale(ch2_rows, clip_min=clip_min, clip_max=clip_max)
    ch3_rows = _clip_and_scale(ch3_rows, clip_min=clip_min, clip_max=clip_max)

    def pad_to_24(rows: np.ndarray) -> np.ndarray:
        """
        Pad an 8×T map to 24×T along the spatial axis.

        We symmetrically add zero rows above and below to reach 24 rows.
        """
        if rows.shape[0] != 8:
            raise ValueError(f"Expected 8 rows before padding, got {rows.shape[0]}")
        
        H, W = rows.shape
        if H >= 24:
            return rows[:24, :]  # Truncate if somehow already larger

        pad_total = 24 - 8
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(rows, ((pad_before, pad_after), (0, 0)), mode="constant")

    # 3) Pad spatial axis to 24 rows
    ch1_24 = pad_to_24(ch1_rows)  # (24, T)
    ch2_24 = pad_to_24(ch2_rows)  # (24, T)
    ch3_24 = pad_to_24(ch3_rows)  # (24, T)

    # 4) Resize time axis to target_width
    ch1_resized = _resize_time_axis(ch1_24, target_width=target_width)  # (24, W)
    ch2_resized = _resize_time_axis(ch2_24, target_width=target_width)
    ch3_resized = _resize_time_axis(ch3_24, target_width=target_width)

    # 5) Stack channels → (3, 24, W)
    img = np.stack([ch1_resized, ch2_resized, ch3_resized], axis=0)  # (3, 24, W)

    return img.astype(np.float32)


def ecg_batch_to_images(
    signals: np.ndarray,
    target_width: int = 2048,
    clip_range: Tuple[float, float] = (-3.0, 3.0),
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Apply ecg_to_contour_image to a batch of ECGs.

    Parameters
    ----------
    signals : np.ndarray
        Batch of ECGs of shape (N, T, 12).
        Each item should already be preprocessed to a fixed duration.
    target_width : int
        Desired temporal size of the resulting images.
    clip_range : Tuple[float, float]
        Clipping range for each channel before scaling to [0, 255].
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.
        Currently implemented sequentially; parallel option for future.

    Returns
    -------
    np.ndarray
        Batch of contour images of shape (N, 3, 24, target_width).
    """
    if signals.ndim != 3 or signals.shape[2] != 12:
        raise ValueError(
            f"signals must have shape (N, T, 12), got {signals.shape}"
        )
    
    N = signals.shape[0]
    if N == 0:
        return np.empty((0, 3, 24, target_width), dtype=np.float32)
    
    # For now, sequential processing
    # Could be parallelized with joblib if needed
    images = []
    for i in range(N):
        img = ecg_to_contour_image(
            signals[i], 
            target_width=target_width, 
            clip_range=clip_range,
            check_normalization=(i == 0)  # Only check first for performance
        )
        images.append(img)
    
    return np.stack(images, axis=0)  # (N, 3, 24, W)