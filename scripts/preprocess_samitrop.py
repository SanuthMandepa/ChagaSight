"""
Preprocess SaMi-Trop ECGs into standardized 1D numpy arrays.

Dataset reference (Zenodo):
- "SaMi-Trop: 12-lead ECG traces with age and mortality annotations"
  Ribeiro et al., 2021, DOI: 10.5281/zenodo.4905618

Raw files (expected layout):
    data/raw/sami_trop/
        exams.csv        # metadata table with column 'exam_id'
        exams.hdf5       # HDF5 file with dataset 'tracings' (unzip exams.zip if needed)

HDF5 layout:
    tracings: shape (1631, 4096, 12)
        - N = 1631 exams (public subset)
        - 4096 samples at 400 Hz (≈10.24 s; some originally 7 s, symmetrically zero-padded)
        - 12 leads in order: [DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]

Dual preprocessing pipeline (aligned with Kim et al. 2025 and Van Santvliet et al. 2025):
    1. Load signal (4096, 12) from HDF5.
    2. Remove symmetric zero-padding (for 7-second recordings).
    3. Gentle baseline removal using moving-average filter (200 ms window).
    4. Resample:
         • To 500 Hz → for 2D image generation (Kim et al.)
         • To 100 Hz → for 1D foundation model (Van Santvliet et al.)
    5. Pad/trim to fixed 10 seconds:
         • 5000 samples at 500 Hz
         • 1000 samples at 100 Hz
    6. Per-lead z-score normalization (only for 500 Hz image path).
    7. Save as:
         data/processed/1d_signals_500hz/sami_trop/<exam_id>.npy   # z-scored, for images
         data/processed/1d_signals_100hz/sami_trop/<exam_id>.npy   # raw amplitudes, for FM

This script generates preprocessed signals used by:
    - scripts/build_images.py                  → 2D RA/LA/LL contour images (from 500 Hz)
    - Future FM pretraining / fine-tuning      → (from 100 Hz)
    - Hybrid alignment experiments
"""

from pathlib import Path
import time
import warnings
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import chagas_label_sami_trop


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
SAMI_ROOT = Path("data/raw/sami_trop")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/sami_trop")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/sami_trop")
METADATA_OUTPUT = Path("data/processed/metadata/sami_trop_processed.csv")

ORIGINAL_FS = 400.0
TARGET_FS_IMAGE = 500.0   # Kim et al. 2025 – for 2D contour images
TARGET_FS_FM = 100.0      # Van Santvliet et al. 2025 – for 1D FM
TARGET_DURATION_SEC = 10.0

TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000

# Gentle baseline removal suitable for SaMi-Trop
BASELINE_METHOD = "moving_average"
BASELINE_KWARGS = {"window_seconds": 0.2}  # 200 ms window


def remove_zero_padding(signal: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Remove symmetric zero-padding from SaMi-Trop signals."""
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end]


def main() -> None:
    start_time = time.time()
    processed_records = []
    failed_records = []

    # Ensure all output directories exist
    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata and HDF5 file
    exams_df = pd.read_csv(SAMI_ROOT / "exams.csv")
    with h5py.File(SAMI_ROOT / "exams.hdf5", "r") as h5_file:
        tracings = h5_file["tracings"]

        for idx, row in tqdm(exams_df.iterrows(), total=len(exams_df), desc="Processing SaMi-Trop"):
            exam_id = int(row["exam_id"])
            try:
                raw_signal = tracings[idx].astype(np.float32)  # (4096, 12)

                # 1. Remove zero-padding
                depadded = remove_zero_padding(raw_signal)

                # 2. Baseline removal
                filtered = remove_baseline(
                    depadded,
                    fs=ORIGINAL_FS,
                    method=BASELINE_METHOD,
                    **BASELINE_KWARGS
                )

                # 3. Image path – 500 Hz + z-score
                signal_500, _ = resample_ecg(filtered, ORIGINAL_FS, TARGET_FS_IMAGE)
                signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
                normalized_500 = zscore_per_lead(signal_500)

                # 4. FM path – 100 Hz, no z-score
                signal_100, _ = resample_ecg(filtered, ORIGINAL_FS, TARGET_FS_FM)
                signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

                # Save
                np.save(OUTPUT_DIR_500 / f"{exam_id}.npy", normalized_500)
                np.save(OUTPUT_DIR_100 / f"{exam_id}.npy", signal_100)

                # Record metadata
                processed_records.append({
                    "exam_id": exam_id,
                    "path_500hz": str(OUTPUT_DIR_500 / f"{exam_id}.npy"),
                    "path_100hz": str(OUTPUT_DIR_100 / f"{exam_id}.npy"),
                    "original_length": raw_signal.shape[0],
                    "depadded_length": depadded.shape[0],
                    "label": chagas_label_sami_trop()
                })

            except Exception as e:
                warnings.warn(f"Failed on exam_id {exam_id} (index {idx}): {e}")
                failed_records.append({
                    "exam_id": exam_id,
                    "index": idx,
                    "error": str(e)
                })

    # Save metadata CSV
    if processed_records:
        metadata_df = pd.DataFrame(processed_records)
        metadata_df.to_csv(METADATA_OUTPUT, index=False)
        print(f"💾 Metadata saved to {METADATA_OUTPUT}")

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ SaMi-Trop preprocessing complete!")
    print(f"   Processed : {len(processed_records)} exams")
    print(f"   Failed    : {len(failed_records)} exams")
    print(f"   Total time: {elapsed_time:.1f} s ({elapsed_time / 60:.1f} min)")
    print(f"   Saved to  :")
    print(f"       Images (500 Hz) → {OUTPUT_DIR_500}")
    print(f"       FM     (100 Hz) → {OUTPUT_DIR_100}")
    print("=" * 60)

    if failed_records:
        print("\n❌ First 10 failed records:")
        for fail in failed_records[:10]:
            print(f"   exam_id {fail['exam_id']} (idx {fail['index']}): {fail['error']}")
        if len(failed_records) > 10:
            print(f"   ... and {len(failed_records) - 10} more")


if __name__ == "__main__":
    main()