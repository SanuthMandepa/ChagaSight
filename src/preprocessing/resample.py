"""
Resampling utilities for ECG signals.

Includes:
    • Polyphase resampling (resample_poly)
    • Symmetric padding and center trimming
    • Standardized time-axis normalization

Used to unify heterogeneous datasets (100/400/500 Hz) into a
common sampling frequency prior to image embedding or model training.

Input:  (T, L)
Output: (T_new, L)
"""

from typing import Tuple, Optional
import warnings

import numpy as np
from scipy.signal import resample_poly


def resample_ecg(
    signal: np.ndarray, 
    fs_in: float, 
    fs_out: float,
    pad_mode: str = 'reflect',
    anti_aliasing: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resample ECG from fs_in to fs_out using polyphase filtering.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    fs_in : float
        Original sampling frequency.
    fs_out : float
        Target sampling frequency.
    pad_mode : str
        Padding mode for edge handling in resample_poly.
    anti_aliasing : bool
        If True, applies anti-aliasing filter (recommended).

    Returns
    -------
    resampled : np.ndarray
        ECG array resampled to fs_out, shape (T_new, leads).
    fs_out : float
        The target sampling frequency (for convenience).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")
    
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"Sampling frequencies must be positive, got fs_in={fs_in}, fs_out={fs_out}")
    
    if fs_in == fs_out:
        return signal.astype(np.float32), fs_out
    
    # Warn about potential aliasing
    if fs_out < fs_in and anti_aliasing:
        nyquist_in = fs_in / 2
        if fs_out < nyquist_in:
            warnings.warn(
                f"Downsampling from {fs_in}Hz to {fs_out}Hz may cause aliasing. "
                f"Consider applying a low-pass filter below {fs_out/2}Hz first."
            )
    
    # Compute up/down factors
    # Use rational approximation for non-integer frequencies
    from fractions import Fraction
    
    # Find rational approximation
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up = frac.numerator
    down = frac.denominator
    
    # Apply resampling
    resampled = resample_poly(
        signal, 
        up=up, 
        down=down, 
        axis=0,
        padtype=pad_mode
    )
    
    # Ensure output length matches expected
    expected_length = int(np.round(signal.shape[0] * fs_out / fs_in))
    if abs(resampled.shape[0] - expected_length) > 1:
        warnings.warn(
            f"Resampled length {resampled.shape[0]} differs from expected "
            f"{expected_length}. Trimming/padding to match."
        )
        resampled = pad_or_trim(resampled, expected_length)
    
    return resampled.astype(np.float32), fs_out


def pad_or_trim(
    signal: np.ndarray, 
    target_length: int,
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad (with zeros) or trim an ECG along the time dimension to a fixed length.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    target_length : int
        Desired number of samples.
    pad_mode : str
        Padding mode: 'constant', 'reflect', 'edge', etc.
    pad_value : float
        Value to use for constant padding.

    Returns
    -------
    np.ndarray
        Padded/trimmed signal of shape (target_length, leads).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")
    
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    T, n_leads = signal.shape

    if T == target_length:
        return signal.astype(np.float32)
    elif T > target_length:
        # Center crop
        start = (T - target_length) // 2
        end = start + target_length
        return signal[start:end, :].astype(np.float32)
    else:
        # Symmetric padding
        pad_total = target_length - T
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        
        if pad_mode == 'constant':
            return np.pad(
                signal, 
                ((pad_before, pad_after), (0, 0)), 
                mode=pad_mode,
                constant_values=pad_value
            ).astype(np.float32)
        else:
            return np.pad(
                signal,
                ((pad_before, pad_after), (0, 0)),
                mode=pad_mode
            ).astype(np.float32)


def resample_and_fix_length(
    signal: np.ndarray,
    fs_in: float,
    fs_out: float,
    target_duration_seconds: float,
    pad_mode: str = 'constant'
) -> Tuple[np.ndarray, float]:
    """
    Resample to target frequency and fix to target duration.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    fs_in : float
        Original sampling frequency.
    fs_out : float
        Target sampling frequency.
    target_duration_seconds : float
        Desired duration in seconds.
    pad_mode : str
        Padding mode for length adjustment.

    Returns
    -------
    np.ndarray
        Resampled and length-adjusted signal.
    float
        Target sampling frequency.
    """
    # First resample
    resampled, _ = resample_ecg(signal, fs_in, fs_out)
    
    # Calculate target length
    target_length = int(np.round(target_duration_seconds * fs_out))
    
    # Pad or trim to target length
    result = pad_or_trim(resampled, target_length, pad_mode=pad_mode)
    
    return result, fs_out