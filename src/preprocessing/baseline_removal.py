"""
Baseline and noise removal utilities for 12-lead ECG signals.

This module provides preprocessing functions commonly used in clinical
ECG research, including:

    • High-pass filtering for baseline wander removal
    • Band-pass filtering for general noise suppression
    • Moving-average baseline estimation

All functions operate on arrays of shape:
        (T, L) → (time samples, number of leads)

These filters form the first step of ECG standardization prior to
resampling and normalization.
"""


from typing import Literal, Optional

import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 0.7,
    order: int = 3,
) -> np.ndarray:
    """
    Remove baseline using a Butterworth high-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, 12) or (T, n_leads).
    fs : float
        Sampling frequency in Hz.
    cutoff_hz : float
        High-pass cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Baseline-corrected ECG, same shape as input.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    nyq = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyq
    
    # Butterworth filter design
    b, a = butter(order, normalized_cutoff, btype="high", analog=False)
    
    # Apply filtfilt for zero-phase filtering
    return filtfilt(b, a, signal, axis=0)


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_cut_hz: float = 0.5,
    high_cut_hz: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """
    ECG band-pass filter (e.g. 0.5–45 Hz), often used in ECG pipelines.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    fs : float
        Sampling frequency in Hz.
    low_cut_hz : float
        Low cutoff frequency in Hz.
    high_cut_hz : float
        High cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Band-pass filtered ECG, same shape as input.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    nyq = 0.5 * fs
    low = low_cut_hz / nyq
    high = high_cut_hz / nyq
    
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass normalized frequencies: low={low}, high={high}. "
            "Check low_cut_hz, high_cut_hz and fs."
        )

    b, a = butter(order, [low, high], btype="band", analog=False)
    return filtfilt(b, a, signal, axis=0)


def moving_average_baseline(
    signal: np.ndarray,
    window_seconds: float,
    fs: float,
) -> np.ndarray:
    """
    Estimate baseline using a moving-average filter and subtract it.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    window_seconds : float
        Window length in seconds for baseline estimation.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Baseline-corrected ECG signal.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (T, leads), got shape {signal.shape}")

    window_samples = int(max(1, round(window_seconds * fs)))
    
    # Ensure window is odd for symmetric convolution
    if window_samples % 2 == 0:
        window_samples += 1
    
    # Create moving average kernel
    kernel = np.ones(window_samples) / window_samples
    
    # Apply convolution to each lead separately with mode='same'
    # Use reflect padding to handle edges
    baseline = np.array([
        np.convolve(signal[:, lead], kernel, mode='same')
        for lead in range(signal.shape[1])
    ]).T
    
    return signal - baseline


def remove_baseline(
    signal: np.ndarray,
    fs: float,
    method: Literal["highpass", "moving_average", "bandpass"] = "highpass",
    **kwargs,
) -> np.ndarray:
    """
    Convenience wrapper to remove baseline / filter ECG with configurable method.

    Parameters
    ----------
    signal : np.ndarray
        ECG array of shape (T, leads).
    fs : float
        Sampling frequency.
    method : {"highpass", "moving_average", "bandpass"}
        Baseline removal / filtering method.
    **kwargs : dict
        Extra args passed to the underlying method.

    Returns
    -------
    np.ndarray
        Filtered ECG.
    """
    # Convert to float32 for numerical stability
    signal = signal.astype(np.float32)
    
    if method == "highpass":
        cutoff_hz = kwargs.get("cutoff_hz", 0.7)
        order = kwargs.get("order", 3)
        return highpass_filter(signal, fs=fs, cutoff_hz=cutoff_hz, order=order)
    
    elif method == "moving_average":
        window_seconds = kwargs.get("window_seconds", 0.8)
        return moving_average_baseline(signal, window_seconds=window_seconds, fs=fs)
    
    elif method == "bandpass":
        low_cut_hz = kwargs.get("low_cut_hz", 0.5)
        high_cut_hz = kwargs.get("high_cut_hz", 45.0)
        order = kwargs.get("order", 4)
        return bandpass_filter(signal, fs=fs, low_cut_hz=low_cut_hz, 
                              high_cut_hz=high_cut_hz, order=order)
    
    else:
        raise ValueError(f"Unknown baseline removal method: {method}")