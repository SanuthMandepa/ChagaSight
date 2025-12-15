"""
Preprocess SaMi-Trop ECGs into standardized 1D numpy arrays.

Dataset reference (Zenodo):
- "SaMi-Trop: 12-lead ECG traces with age and mortality annotations"
  Ribeiro et al., 2021, DOI: 10.5281/zenodo.4905618

Raw files (expected layout):
    data/raw/sami_trop/
        exams.csv        # metadata table with column 'exam_id'
        exams.hdf5       # HDF5 file with dataset 'tracings'

HDF5 layout:
    tracings: shape (N, 4096, 12)
        - N = number of exams (1631 in the public subset)
        - 4096 samples at 400 Hz (≈10.24 s; some originally 7 s, zero-padded)
        - 12 leads in order: [DI, DII, DIII, AVR, AVL, AVF, V1..V6]

Standardized preprocessing pipeline (aligned with PTB-XL):
    1. Load signal (4096, 12) from HDF5.
    2. Baseline removal using moving-average filter.
    3. Resample to TARGET_FS (default 400 Hz).
    4. Pad/trim to TARGET_DURATION_SEC (default 10 s → 4000 samples).
    5. Per-lead z-score normalization.
    6. Save as:
        data/processed/1d_signals/sami_trop/<exam_id>.npy

This script is intended to be run once to generate the processed 1D signals
for SaMi-Trop, which are then used by:
    - scripts/build_images.py        → 2D RA/LA/LL contour images
    - src/dataloaders/fm_signal_dataset.py
    - src/dataloaders/image_dataset.py
"""

from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SAMI_ROOT = Path("data/raw/sami_trop")
OUTPUT_DIR = Path("data/processed/1d_signals/sami_trop")

# SaMi-Trop tracings are already at 400 Hz, but we keep this configurable
TARGET_FS = 400.0
TARGET_DURATION_SEC = 10.0  # to match PTB-XL (10 s) and CODE-15%


# -------------------------------------------------------------------------
# Preprocessing for a single SaMi-Trop exam
# -------------------------------------------------------------------------


def preprocess_single_exam(
    signal: np.ndarray,
    fs_in: float = 400.0,
    fs_target: float = TARGET_FS,
    duration_sec: float = TARGET_DURATION_SEC,
    baseline_window_sec: float = 0.8,
) -> np.ndarray:
    """
    Preprocess a single SaMi-Trop ECG recording.

    Steps:
        1) Baseline removal via moving-average filter (paper 2 style).
        2) Resample to fs_target using polyphase resampling.
        3) Pad or trim to fixed duration.
        4) Per-lead z-score normalization.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal array of shape (T, 12).
    fs_in : float
        Original sampling frequency (SaMi-Trop uses 400 Hz).
    fs_target : float
        Target sampling frequency (default 400 Hz).
    duration_sec : float
        Target fixed duration (default 10 s).
    baseline_window_sec : float
        Window (in seconds) for moving-average baseline estimation.

    Returns
    -------
    np.ndarray
        Preprocessed ECG of shape (T_fixed, 12), dtype float32.
    """
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(
            f"Expected signal of shape (T, 12), got {signal.shape}"
        )

    # Ensure float32 for numerical stability and storage efficiency
    signal = signal.astype(np.float32)

    # 1) Baseline removal with moving-average filter
    signal = remove_baseline(
        signal,
        fs=fs_in,
        method="moving_average",
        window_seconds=baseline_window_sec,
    )

    # 2) Resample (if fs_in != fs_target, otherwise this is a no-op)
    signal, fs_rs = resample_ecg(signal, fs_in=fs_in, fs_out=fs_target)

    # 3) Pad or trim to fixed length
    target_len = int(round(duration_sec * fs_rs))
    signal = pad_or_trim(signal, target_length=target_len)  # (T_fixed, 12)

    # 4) Per-lead z-score normalization
    signal = zscore_per_lead(signal)

    return signal.astype(np.float32)


# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------


def main():
    # 1) Load metadata CSV
    csv_path = SAMI_ROOT / "exams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find SaMi-Trop CSV at: {csv_path}")

    print(f"📄 Loading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic sanity check
    if "exam_id" not in df.columns:
        raise ValueError("Expected column 'exam_id' in exams.csv")

    # 2) Open HDF5 file with tracings
    h5_path = SAMI_ROOT / "exams.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find SaMi-Trop HDF5 at: {h5_path}")

    print(f"📂 Opening HDF5 tracings from: {h5_path}")
    h5_file = h5py.File(h5_path, "r")

    if "tracings" not in h5_file:
        raise KeyError("HDF5 file does not contain dataset 'tracings'")

    tracings = h5_file["tracings"]  # shape (N, 4096, 12)
    num_records_h5 = tracings.shape[0]
    num_records_csv = len(df)

    print(f"🔢 HDF5 tracings shape: {tracings.shape}")
    print(f"🔢 CSV rows: {num_records_csv}")

    if num_records_h5 != num_records_csv:
        print(
            f"⚠️ Warning: HDF5 has {num_records_h5} records but CSV has {num_records_csv} rows."
            " Assuming 1:1 alignment by index."
        )

    # 3) Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving preprocessed signals to: {OUTPUT_DIR}")

    # 4) Loop over all exams
    for idx, row in df.iterrows():
        exam_id = int(row["exam_id"])
        # tracings[idx] is (4096, 12)
        raw_signal = tracings[idx, :, :]  # (T, 12)

        if idx % 100 == 0:
            print(
                f"[{idx+1}/{num_records_csv}] Processing exam_id={exam_id} "
                f"(raw shape: {raw_signal.shape})"
            )

        try:
            processed = preprocess_single_exam(
                signal=raw_signal,
                fs_in=400.0,
                fs_target=TARGET_FS,
                duration_sec=TARGET_DURATION_SEC,
                baseline_window_sec=0.8,
            )
        except Exception as e:
            print(f"❌ Failed to process exam_id={exam_id}: {e}")
            continue

        out_path = OUTPUT_DIR / f"{exam_id}.npy"
        np.save(out_path, processed)

    # 5) Close HDF5 file
    h5_file.close()

    print("✅ SaMi-Trop preprocessing complete.")


if __name__ == "__main__":
    main()