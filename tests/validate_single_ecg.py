# tests/validate_single_ecg.py
"""
Visual sanity-check for ONE ECG across the full pipeline:
raw (WFDB) → baseline-removed → resampled (500 Hz & 100 Hz) → fixed-length → z-scored (500 Hz only) → 2D image.

Supported datasets:
- ptbxl     : WFDB records under data/official_wfdb/ptbxl/
- sami_trop : WFDB records under data/official_wfdb/sami_trop/
- code15    : WFDB records under data/official_wfdb/code15/

Usage:
python tests/validate_single_ecg.py --dataset code15 --id 113
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)

def load_wfdb_signal(dataset, exam_id):
    wfdb_dir = Path("data/official_wfdb") / dataset
    rec_path = wfdb_dir / str(exam_id)
    if dataset == 'ptbxl':
        # PTB-XL has subfolders (e.g., 00000/00001_hr)
        for root, _, files in os.walk(wfdb_dir):
            if f"{exam_id}.hea" in files:
                rec_path = Path(root) / exam_id
                break

    if not rec_path.with_suffix('.hea').exists():
        raise FileNotFoundError(f"WFDB record not found: {rec_path}")

    signal, fields = wfdb.rdsamp(str(rec_path))
    fs = fields['fs']
    return signal, fs, fields

def validate_single_ecg(dataset, exam_id):
    output_dir = Path("tests/verification_outputs/full")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from WFDB
    raw_signal, raw_fs, metadata = load_wfdb_signal(dataset, exam_id)
    print(f"Raw signal shape: {raw_signal.shape}, fs: {raw_fs}")

    # Baseline removal (dataset-specific)
    if dataset == 'ptbxl':
        baseline_method = 'bandpass'
    elif dataset == 'code15':
        baseline_method = None  # Already filtered
    else:
        baseline_method = 'moving_average'
    
    if baseline_method:
        baseline_removed = remove_baseline(raw_signal, raw_fs, baseline_method)
    else:
        baseline_removed = raw_signal

    # Resample to 500 Hz for images
    signal_500, _ = resample_ecg(baseline_removed, raw_fs, TARGET_FS_IMAGE)
    signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
    
    # Normalize and clip for images
    normalized_500 = normalize_dataset(signal_500)
    clipped_500 = np.clip(normalized_500, *CLIP_RANGE)

    # Generate 2D image
    img = ecg_to_contour_image(clipped_500, target_width=TARGET_WIDTH, clip_range=CLIP_RANGE)
    print(f"2D image shape: {img.shape}")

    # Resample to 100 Hz for FM (no normalize/clip)
    signal_100, _ = resample_ecg(baseline_removed, raw_fs, TARGET_FS_FM)
    signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

    # Plots
    t_raw = np.arange(raw_signal.shape[0]) / raw_fs
    t_500 = np.arange(signal_500.shape[0]) / TARGET_FS_IMAGE
    t_100 = np.arange(signal_100.shape[0]) / TARGET_FS_FM

    fig, axs = plt.subplots(5, 1, figsize=(12, 20))
    axs[0].plot(t_raw, raw_signal[:, 0], label='Raw')
    axs[0].set_title('Raw Signal (Lead I)')
    axs[1].plot(t_raw, baseline_removed[:, 0], label='Baseline Removed')
    axs[1].set_title('Baseline Removed')
    axs[2].plot(t_500, signal_500[:, 0], label='Resampled 500 Hz')
    axs[2].set_title('Resampled 500 Hz')
    axs[3].plot(t_500, normalized_500[:, 0], label='Normalized')
    axs[3].set_title('Normalized 500 Hz')
    axs[4].plot(t_100, signal_100[:, 0], label='Resampled 100 Hz')
    axs[4].set_title('Resampled 100 Hz (for FM)')
    for ax in axs:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_{exam_id}_pipeline.png")
    plt.close()

    # 2D image plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img[0], cmap='gray')
    axs[0].set_title('RA Contour')
    axs[1].imshow(img[1], cmap='gray')
    axs[1].set_title('LA Contour')
    axs[2].imshow(img[2], cmap='gray')
    axs[2].set_title('LL Contour')
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_{exam_id}_2d_image.png")
    plt.close()

    print(f"Plots saved for {dataset} ID {exam_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate full pipeline for a single record.")
    parser.add_argument("--dataset", required=True, choices=['ptbxl', 'sami_trop', 'code15'])
    parser.add_argument("--id", type=str, required=True)  # Changed to str for PTB-XL IDs
    args = parser.parse_args()
    validate_single_ecg(args.dataset, args.id)