"""2D Image embedding for ECG signals (all bugs fixed)."""

import numpy as np
from scipy.interpolate import interp1d


def build_2d_image(signal, target_height=24, target_width=2048):
    """Build 2D image from ECG signal using RA/LA/LL contour-like grouping.

    Args:
        signal: (12, num_samples) Z-score normalized signal
                Expected range: roughly [-3, 3] after clipping
        target_height: Image height (24 = 8 rows per group × 3 groups)
        target_width: Image width (2048)

    Returns:
        img: (3, 24, 2048) uint8 image in [0, 255]
    """
    num_leads, num_samples = signal.shape

    # Lead groups (roughly RA/LA/LL-related):
    #   Group 0: I, II, III
    #   Group 1: aVR, aVL, aVF
    #   Group 2: V1–V6
    lead_groups = [
        [0, 1, 2],            # I, II, III
        [3, 4, 5],            # aVR, aVL, aVF
        [6, 7, 8, 9, 10, 11], # V1–V6
    ]

    img = np.zeros((3, target_height, target_width), dtype=np.float32)

    # Interpolate to fixed width
    if num_samples != target_width:
        x_old = np.linspace(0, 1, num_samples)
        x_new = np.linspace(0, 1, target_width)
        signal_interp = np.zeros((num_leads, target_width), dtype=np.float32)
        for lead_idx in range(num_leads):
            f = interp1d(x_old, signal[lead_idx], kind="linear")
            signal_interp[lead_idx] = f(x_new)
        signal = signal_interp

    rows_per_group = target_height // 3  # 24 / 3 = 8

    # Fill each channel with its group of leads, tiled over rows
    for channel_idx, lead_group in enumerate(lead_groups):
        base_row = channel_idx * rows_per_group
        rows_per_lead = max(1, rows_per_group // len(lead_group))

        for i, lead_idx in enumerate(lead_group):
            if lead_idx >= num_leads:
                continue

            start_row = base_row + i * rows_per_lead
            end_row = min(base_row + rows_per_group, start_row + rows_per_lead)

            img[channel_idx, start_row:end_row, :] = signal[lead_idx, :]

    # Clip to [-3, 3] then scale to [0, 255]
    img = np.clip(img, -3.0, 3.0)
    img = (img + 3.0) / 6.0 * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
