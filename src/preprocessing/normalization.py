# normalization.py
# Per-lead normalization. Input shape: (L, T)

from __future__ import annotations

from typing import Optional
import numpy as np


def normalize_per_lead(
    signal: np.ndarray,
    method: str = "zscore",
    clip_std: Optional[float] = 3.0,
    eps: float = 1e-8,
) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (L,T), got shape {signal.shape}")

    x = signal.astype(np.float32, copy=False)
    out = np.zeros_like(x, dtype=np.float32)

    for i in range(x.shape[0]):
        lead = x[i]
        if method == "zscore":
            mu = float(np.mean(lead))
            sd = float(np.std(lead))
            if sd < 1e-6:
                out[i] = 0.0
            else:
                y = (lead - mu) / (sd + eps)
                if clip_std is not None:
                    y = np.clip(y, -float(clip_std), float(clip_std))
                out[i] = y.astype(np.float32, copy=False)
        elif method == "minmax":
            mn = float(np.min(lead))
            mx = float(np.max(lead))
            if (mx - mn) < 1e-6:
                out[i] = 0.0
            else:
                out[i] = ((lead - mn) / (mx - mn)).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return out
