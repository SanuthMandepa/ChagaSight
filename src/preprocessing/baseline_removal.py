# baseline_removal.py
# Baseline removal / filtering utilities for ECG (12-lead).
# Supports input shape (L, T) where L=12 leads.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import numpy as np
from scipy.signal import butter, filtfilt


Method = Literal["bandpass", "highpass", "movingaverage", "none"]


@dataclass(frozen=True)
class BaselineConfig:
    method: Method = "bandpass"
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    cutoff_hz: float = 0.5
    order: int = 4
    ma_window_seconds: float = 0.2


def _ensure_2d_lead_first(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"signal must be 2D, got {x.ndim}D with shape {x.shape}")
    return x


def remove_baseline_bandpass(
    signal: np.ndarray,
    fs: float,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    signal = _ensure_2d_lead_first(signal).astype(np.float32, copy=False)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyq = 0.5 * fs
    low = lowcut_hz / nyq
    high = highcut_hz / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid bandpass: low={lowcut_hz}, high={highcut_hz}, fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    out = np.zeros_like(signal, dtype=np.float32)
    for i in range(signal.shape[0]):
        out[i] = filtfilt(b, a, signal[i]).astype(np.float32, copy=False)
    return out


def remove_baseline_highpass(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 0.5,
    order: int = 3,
) -> np.ndarray:
    signal = _ensure_2d_lead_first(signal).astype(np.float32, copy=False)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyq = 0.5 * fs
    cutoff = cutoff_hz / nyq
    if not (0.0 < cutoff < 1.0):
        raise ValueError(f"Invalid highpass: cutoff={cutoff_hz}, fs={fs}")
    b, a = butter(order, cutoff, btype="high")
    out = np.zeros_like(signal, dtype=np.float32)
    for i in range(signal.shape[0]):
        out[i] = filtfilt(b, a, signal[i]).astype(np.float32, copy=False)
    return out


def remove_baseline_moving_average(
    signal: np.ndarray,
    fs: float,
    window_seconds: float = 0.2,
) -> np.ndarray:
    signal = _ensure_2d_lead_first(signal).astype(np.float32, copy=False)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    win = int(round(window_seconds * fs))
    win = max(1, win)
    kernel = np.ones(win, dtype=np.float32) / float(win)

    out = np.zeros_like(signal, dtype=np.float32)
    for i in range(signal.shape[0]):
        baseline = np.convolve(signal[i], kernel, mode="same").astype(np.float32, copy=False)
        out[i] = (signal[i] - baseline).astype(np.float32, copy=False)
    return out


def remove_baseline(
    signal: np.ndarray,
    fs: float,
    config: Optional[BaselineConfig] = None,
    **override: Any,
) -> np.ndarray:
    """
    Top-level wrapper. Input: (L, T). Output: (L, T), float32.
    """
    if config is None:
        config = BaselineConfig()

    method: Method = override.get("method", config.method)

    if method in ("none",):
        return _ensure_2d_lead_first(signal).astype(np.float32, copy=False)

    if method == "bandpass":
        return remove_baseline_bandpass(
            signal,
            fs=fs,
            lowcut_hz=override.get("lowcut_hz", config.lowcut_hz),
            highcut_hz=override.get("highcut_hz", config.highcut_hz),
            order=int(override.get("order", config.order)),
        )
    if method == "highpass":
        return remove_baseline_highpass(
            signal,
            fs=fs,
            cutoff_hz=override.get("cutoff_hz", config.cutoff_hz),
            order=int(override.get("order", 3)),
        )
    if method == "movingaverage":
        return remove_baseline_moving_average(
            signal,
            fs=fs,
            window_seconds=override.get("ma_window_seconds", config.ma_window_seconds),
        )

    raise ValueError(f"Unknown baseline method: {method}")
