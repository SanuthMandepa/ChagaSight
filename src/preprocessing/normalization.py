"""Normalization utilities for ECG signals."""

import numpy as np


def normalize_per_lead(signal, method="zscore", clip_std=3.0):
    """Normalize each lead independently.

    Args:
        signal: (num_leads, num_samples) ECG signal
        method: 'zscore' or 'minmax'
        clip_std: For zscore, optional clipping to [-clip_std, clip_std]

    Returns:
        Normalized signal as float32
    """
    normalized = np.zeros_like(signal, dtype=np.float32)

    for lead_idx in range(signal.shape[0]):
        lead_signal = signal[lead_idx].astype(np.float32)

        if method == "zscore":
            mean = np.mean(lead_signal)
            std = np.std(lead_signal)

            if std < 1e-6:  # Avoid division by zero
                normalized[lead_idx] = 0.0
            else:
                normalized[lead_idx] = (lead_signal - mean) / std

                if clip_std is not None:
                    normalized[lead_idx] = np.clip(
                        normalized[lead_idx], -clip_std, clip_std
                    )

        elif method == "minmax":
            min_val = np.min(lead_signal)
            max_val = np.max(lead_signal)

            if max_val - min_val < 1e-6:
                normalized[lead_idx] = 0.0
            else:
                normalized[lead_idx] = (lead_signal - min_val) / (max_val - min_val)

        else:
            raise ValueError(f"Unknown method: {method}")

    return normalized
