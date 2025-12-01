"""
Resampling utilities for ECG signals.
"""

from typing import Tuple

import numpy as np
from scipy.signal import resample_poly


def resample_ecg(signal: np.ndarray, fs_in: float, fs_out: float) -> Tuple[np.ndarray, float]:
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

    Returns
    -------
    resampled : np.ndarray
        ECG array resampled to fs_out, shape (T_new, leads).
    fs_out : float
        The target sampling frequency (for convenience).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    if fs_in == fs_out:
        return signal, fs_out

    # Compute up/down factors as integers if possible
    gcd = np.gcd(int(fs_in), int(fs_out))
    up = int(fs_out // gcd)
    down = int(fs_in // gcd)

    resampled = resample_poly(signal, up=up, down=down, axis=0)
    return resampled, fs_out


def pad_or_trim(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad (with zeros) or trim an ECG along the time dimension to a fixed length.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    target_length : int
        Desired number of samples.

    Returns
    -------
    np.ndarray
        Padded/trimmed signal of shape (target_length, leads).
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    T, n_leads = signal.shape

    if T == target_length:
        return signal
    elif T > target_length:
        # Center crop
        start = (T - target_length) // 2
        end = start + target_length
        return signal[start:end, :]
    else:
        # Symmetric zero padding
        pad_total = target_length - T
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(signal, ((pad_before, pad_after), (0, 0)), mode="constant")
