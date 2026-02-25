# image_embedding.py
# Build 2D 3-channel "ECG image" tensors from 12-lead ECG.
# Output: (3, 24, 2048) uint8.
#
# Notes:
# - Input expected to be normalized already (typically z-score, clipped).
# - Input shape: (12, T) at 500 Hz, T ~ 5000 after pad/trim to 10 seconds.
# - We crop/choose exactly 2048 samples (≈4.096 sec at 500 Hz) for the 2D embedding.

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _to_uint8_from_clipped(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -clip, clip)
    x = (x + clip) / (2.0 * clip)  # [0,1]
    x = (255.0 * x).round()
    return x.astype(np.uint8)


def _stack_leads_to_height24(leads12: np.ndarray) -> np.ndarray:
    """
    leads12: (12, W) float
    Returns: (24, W) float by duplicating each lead into 2 rows.
    """
    if leads12.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got {leads12.shape[0]}")
    W = leads12.shape[1]
    out = np.zeros((24, W), dtype=np.float32)
    for i in range(12):
        out[2 * i + 0] = leads12[i]
        out[2 * i + 1] = leads12[i]
    return out


def _compute_wct_variant(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    signals: (12, W) in STANDARD_12 order.
    Returns 3 variants (each 12xW):
      ch0: original
      ch1: WCT-adjusted precordials (V1-V6) by adding WCT=(I+II)/3
      ch2: common-mode removed by subtracting WCT from all leads
    """
    s = signals.astype(np.float32, copy=False)

    lead_I = s[0]
    lead_II = s[1]
    wct = (lead_I + lead_II) / 3.0  # simple approximation from limb leads

    ch0 = s

    ch1 = s.copy()
    # Add WCT to V1..V6 (indices 6..11) to approximate electrode potentials
    ch1[6:12] = ch1[6:12] + wct

    ch2 = s - wct  # remove common mode-ish component

    return ch0, ch1, ch2


def build_2d_image(
    signal_12lead_500hz: np.ndarray,
    target_width: int = 2048,
    random_crop: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Input: (12, T) float32, normalized.
    Output: (3, 24, 2048) uint8.
    """
    x = np.asarray(signal_12lead_500hz, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 12:
        raise ValueError(f"Expected shape (12,T), got {x.shape}")

    T = x.shape[1]
    if T < target_width:
        raise ValueError(f"Signal too short for target_width={target_width}: T={T}")

    if random_crop:
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, T - target_width + 1))
    else:
        start = (T - target_width) // 2

    crop = x[:, start : start + target_width]  # (12, 2048)

    ch0, ch1, ch2 = _compute_wct_variant(crop)

    img0 = _to_uint8_from_clipped(_stack_leads_to_height24(ch0))
    img1 = _to_uint8_from_clipped(_stack_leads_to_height24(ch1))
    img2 = _to_uint8_from_clipped(_stack_leads_to_height24(ch2))

    out = np.stack([img0, img1, img2], axis=0)  # (3,24,2048)
    return out.astype(np.uint8, copy=False)
