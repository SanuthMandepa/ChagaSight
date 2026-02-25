"""Resampling utilities for ECG signals."""

import numpy as np
from scipy.signal import resample


def resample_signal(signal, original_fs, target_fs):
    """Resample ECG signal to target frequency.

    Args:
        signal: (num_leads, num_samples) ECG signal
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        Resampled signal (num_leads, new_num_samples)
    """
    if original_fs == target_fs:
        return signal

    # Calculate new number of samples
    original_num_samples = signal.shape[1]
    new_num_samples = int(original_num_samples * target_fs / original_fs)

    # Resample each lead
    resampled = np.zeros((signal.shape[0], new_num_samples))
    for lead_idx in range(signal.shape[0]):
        resampled[lead_idx] = resample(signal[lead_idx], new_num_samples)

    return resampled


def pad_or_trim(signal, target_length):
    """Pad or trim signal to the target length.

    Args:
        signal: (num_leads, num_samples)
        target_length: Desired number of samples

    Returns:
        Signal with target_length samples
    """
    current_length = signal.shape[1]

    if current_length == target_length:
        return signal
    elif current_length < target_length:
        # Pad with zeros at the end
        pad_width = target_length - current_length
        return np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
    else:
        # Trim from center
        start = (current_length - target_length) // 2
        return signal[:, start : start + target_length]
