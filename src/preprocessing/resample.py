# resample.py
# Resampling + pad/trim utilities. Input shape: (L, T)

from __future__ import annotations

import numpy as np
from scipy.signal import resample


def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (L,T), got shape {signal.shape}")
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Sampling rates must be positive, got original_fs={original_fs}, target_fs={target_fs}")

    if float(original_fs) == float(target_fs):
        return signal.astype(np.float32, copy=False)

    old_len = signal.shape[1]
    new_len = int(round(old_len * float(target_fs) / float(original_fs)))
    new_len = max(1, new_len)

    out = np.zeros((signal.shape[0], new_len), dtype=np.float32)
    for i in range(signal.shape[0]):
        out[i] = resample(signal[i].astype(np.float32, copy=False), new_len).astype(np.float32, copy=False)
    return out


def pad_or_trim(signal: np.ndarray, target_len: int) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (L,T), got shape {signal.shape}")

    L, T = signal.shape
    if T == target_len:
        return signal.astype(np.float32, copy=False)

    if T < target_len:
        pad = target_len - T
        out = np.pad(signal, ((0, 0), (0, pad)), mode="constant")
        return out.astype(np.float32, copy=False)

    start = (T - target_len) // 2
    out = signal[:, start : start + target_len]
    return out.astype(np.float32, copy=False)
