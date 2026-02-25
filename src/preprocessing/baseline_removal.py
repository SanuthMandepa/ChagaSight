"""
Baseline removal for ECG signals.
CORRECTED VERSION - All bugs fixed.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def remove_baseline_bandpass(signal, fs=500, low_cut_hz=0.5, high_cut_hz=40.0, order=4):
    """Remove baseline using bandpass filter.

    Args:
        signal: (num_leads, num_samples) ECG signal
        fs: Sampling frequency in Hz
        low_cut_hz: High-pass cutoff (removes baseline wander)
        high_cut_hz: Low-pass cutoff (removes high-freq noise)
        order: Filter order

    Returns:
        Filtered signal with same shape
    """
    nyquist = fs / 2.0
    low = low_cut_hz / nyquist
    high = high_cut_hz / nyquist

    # Create bandpass filter
    b, a = butter(order, [low, high], btype="band")

    # Apply filter to each lead
    filtered = np.zeros_like(signal)
    for lead_idx in range(signal.shape[0]):
        filtered[lead_idx] = filtfilt(b, a, signal[lead_idx])

    return filtered


def remove_baseline_highpass(signal, fs=500, cutoff_hz=0.5, order=3):
    """Remove baseline using highpass filter (better preserves P- and T-waves).

    Args:
        signal: (num_leads, num_samples) ECG signal
        fs: Sampling frequency
        cutoff_hz: Cutoff frequency
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    cutoff = cutoff_hz / nyquist

    # Create highpass filter
    b, a = butter(order, cutoff, btype="high")

    # Apply filter
    filtered = np.zeros_like(signal)
    for lead_idx in range(signal.shape[0]):
        filtered[lead_idx] = filtfilt(b, a, signal[lead_idx])

    return filtered


def remove_baseline_moving_average(signal, fs=500, window_seconds=0.2):
    """Remove baseline using moving-average subtraction.

    Args:
        signal: (num_leads, num_samples)
        fs: Sampling frequency
        window_seconds: Window size in seconds

    Returns:
        Signal with baseline removed
    """
    window_samples = int(window_seconds * fs)

    filtered = np.zeros_like(signal)
    for lead_idx in range(signal.shape[0]):
        # Compute moving average
        kernel = np.ones(window_samples) / window_samples
        baseline = np.convolve(signal[lead_idx], kernel, mode="same")

        # Subtract baseline
        filtered[lead_idx] = signal[lead_idx] - baseline

    return filtered


def remove_baseline(signal, fs=500, method="bandpass", **kwargs):
    """Top-level baseline removal wrapper.

    Args:
        signal: (num_leads, num_samples) ECG signal
        fs: Sampling frequency
        method: 'bandpass', 'highpass', 'moving_average', or None
        **kwargs: Method-specific parameters

    Returns:
        Signal with baseline removed
    """
    if method == "bandpass":
        return remove_baseline_bandpass(signal, fs=fs, **kwargs)
    elif method == "highpass":
        return remove_baseline_highpass(signal, fs=fs, **kwargs)
    elif method == "moving_average":
        return remove_baseline_moving_average(signal, fs=fs, **kwargs)
    elif method is None or method == "none":
        return signal  # No filtering
    else:
        raise ValueError(f"Unknown method: {method}")
