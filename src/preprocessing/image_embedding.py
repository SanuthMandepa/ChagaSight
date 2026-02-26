# image_embedding.py - CORRECTED VERSION WITH PROPER WCT
# Build 2D 3-channel "ECG image" tensors from 12-lead ECG.
# Output: (3, 24, 2048) uint8.
#
# CORRECTED: Proper Wilson's Central Terminal (WCT) re-referencing
# Following Kim et al. 2025 Section 2.2

from __future__ import annotations

from typing import Optional
import numpy as np


STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _to_uint8_from_clipped(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Convert clipped float to uint8 [0, 255]"""
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


def _compute_wct_proper(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CORRECTED: Proper WCT re-referencing following Kim et al. 2025.
    
    Standard 12-lead ECG:
    - Limb leads: I=LA-RA, II=LL-RA, III=LL-LA
    - Augmented: aVR, aVL, aVF (referenced to WCT)
    - Precordial: V1-V6 (referenced to WCT)
    
    Args:
        signals: (12, W) in STANDARD_12 order
                [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    
    Returns:
        ch0: (12, W) RA-referenced view
        ch1: (12, W) LA-referenced view
        ch2: (12, W) LL-referenced view
    """
    s = signals.astype(np.float32, copy=False)
    
    # Extract standard leads
    I = s[0]    # LA - RA
    II = s[1]   # LL - RA
    III = s[2]  # LL - LA
    aVR = s[3]
    aVL = s[4]
    aVF = s[5]
    V1, V2, V3, V4, V5, V6 = s[6:12]
    
    # Compute limb potentials (set RA = 0 as reference)
    # From: I = LA - RA, II = LL - RA, III = LL - LA
    RA = np.zeros_like(I)
    LA = I   # Since I = LA - RA and RA = 0
    LL = II  # Since II = LL - RA and RA = 0
    
    # Wilson's Central Terminal
    WCT = (RA + LA + LL) / 3.0
    
    # Channel 0: RA-referenced view
    # All leads expressed relative to RA
    ch0 = np.stack([
        LA - RA,           # I
        LL - RA,           # II
        LL - LA,           # III (unchanged)
        RA - WCT,          # aVR re-referenced
        LA - WCT,          # aVL re-referenced
        LL - WCT,          # aVF re-referenced
        V1 - RA + WCT,     # V1 re-referenced from WCT to RA
        V2 - RA + WCT,
        V3 - RA + WCT,
        V4 - RA + WCT,
        V5 - RA + WCT,
        V6 - RA + WCT
    ])  # (12, W)
    
    # Channel 1: LA-referenced view
    ch1 = np.stack([
        RA - LA,           # -I
        LL - LA,           # III
        LL - LA,           # III (same)
        RA - WCT,          # aVR
        LA - WCT,          # aVL
        LL - WCT,          # aVF
        V1 - LA + WCT,     # V1 re-referenced from WCT to LA
        V2 - LA + WCT,
        V3 - LA + WCT,
        V4 - LA + WCT,
        V5 - LA + WCT,
        V6 - LA + WCT
    ])  # (12, W)
    
    # Channel 2: LL-referenced view
    ch2 = np.stack([
        LA - LL,           # -III
        RA - LL,           # -II
        LA - LL,           # -III (same)
        RA - WCT,          # aVR
        LA - WCT,          # aVL
        LL - WCT,          # aVF
        V1 - LL + WCT,     # V1 re-referenced from WCT to LL
        V2 - LL + WCT,
        V3 - LL + WCT,
        V4 - LL + WCT,
        V5 - LL + WCT,
        V6 - LL + WCT
    ])  # (12, W)
    
    return ch0, ch1, ch2


def build_2d_image(
    signal_12lead_500hz: np.ndarray,
    target_width: int = 2048,
    random_crop: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build 2D image with CORRECTED proper WCT re-referencing.
    
    Input: (12, T) float32, normalized and clipped to [-3, 3]
    Output: (3, 24, 2048) uint8
    
    Paper: Kim et al. (2025) Section 2.2 - 2D Image Construction
    """
    x = np.asarray(signal_12lead_500hz, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 12:
        raise ValueError(f"Expected shape (12,T), got {x.shape}")

    T = x.shape[1]
    if T < target_width:
        raise ValueError(f"Signal too short for target_width={target_width}: T={T}")

    # Crop to target width
    if random_crop:
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, T - target_width + 1))
    else:
        start = (T - target_width) // 2

    crop = x[:, start : start + target_width]  # (12, 2048)

    # CORRECTED: Use proper WCT re-referencing
    ch0, ch1, ch2 = _compute_wct_proper(crop)  # Each: (12, 2048)
    
    # Convert each channel to (24, 2048) uint8
    img0 = _to_uint8_from_clipped(_stack_leads_to_height24(ch0))
    img1 = _to_uint8_from_clipped(_stack_leads_to_height24(ch1))
    img2 = _to_uint8_from_clipped(_stack_leads_to_height24(ch2))
    
    # Stack into (3, 24, 2048)
    out = np.stack([img0, img1, img2], axis=0)
    return out.astype(np.uint8, copy=False)