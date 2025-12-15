"""
Preprocess PTB-XL into standardized 1D numpy arrays.

Pipeline:
    1. Load WFDB ECG record (records100 or records500)
    2. Apply baseline / band-pass filtering (0.5–45 Hz)
    3. Resample to target sampling rate (default: 400 Hz)
    4. Pad or trim to fixed duration (default: 10 seconds)
    5. Per-lead z-score normalization
    6. Save each sample as: data/processed/1d_signals/ptbxl/<ecg_id>.npy
    7. Save metadata CSV for reproducibility

This script forms the foundation for all downstream image conversion
and model training steps.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
PTBXL_ROOT = Path("data/raw/ptbxl")
OUTPUT_DIR = Path("data/processed/1d_signals/ptbxl")

# Target sampling rate and duration (used across all datasets)
TARGET_FS = 400.0
TARGET_DURATION_SEC = 10.0

# Use low-res 100 Hz WFDB files (filename_lr) or 500 Hz (filename_hr)
USE_LOW_RES = True  # recommended: True


# -------------------------------------------------------------------------
# Preprocessing for a single record
# -------------------------------------------------------------------------
def preprocess_single_record(
    ecg_path: Path,
    bandpass: bool = True,
    fs_target: float = TARGET_FS,
    duration_sec: float = TARGET_DURATION_SEC,
) -> np.ndarray:
    """
    Load → filter → resample → pad/trim → z-score normalize → return array.

    Parameters
    ----------
    ecg_path : Path
        Path to the WFDB record without extension (e.g. data/raw/ptbxl/records100/00000/00001_lr)
    bandpass : bool
        If True, apply 0.5–45 Hz band-pass filter.
    fs_target : float
        Target sampling frequency in Hz.
    duration_sec : float
        Target fixed duration in seconds.

    Returns
    -------
    np.ndarray
        Preprocessed ECG with shape (T, leads) and dtype float32.
    """
    # wfdb.rdrecord handles ".dat"/".hea" internally
    record = wfdb.rdrecord(str(ecg_path))
    signal = record.p_signal.astype(np.float32)  # shape (T, 12)
    fs_in = float(record.fs)

    # 1) Baseline / band-pass filtering
    if bandpass:
        signal = remove_baseline(
            signal,
            fs=fs_in,
            method="bandpass",
            low_cut_hz=0.5,
            high_cut_hz=45.0,
            order=4,
        )
    else:
        # Fallback: simple high-pass if needed
        signal = remove_baseline(signal, fs=fs_in, method="highpass", cutoff_hz=0.7, order=3)

    # 2) Resample to target_fs
    signal, fs_rs = resample_ecg(signal, fs_in=fs_in, fs_out=fs_target)

    # 3) Pad or trim to fixed duration
    target_len = int(round(duration_sec * fs_rs))
    signal = pad_or_trim(signal, target_length=target_len)  # (T_fixed, leads)

    # 4) Per-lead z-score normalization (using the function from normalization.py)
    signal = zscore_per_lead(signal)

    return signal.astype(np.float32)


# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------
def main():
    print("🔍 Loading PTB-XL metadata...")
    db_path = PTBXL_ROOT / "ptbxl_database.csv"
    df = pd.read_csv(db_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select which column to use for WFDB filenames
    filename_col = "filename_lr" if USE_LOW_RES else "filename_hr"

    for idx, row in df.iterrows():
        ecg_id = int(row["ecg_id"])
        rel_path = row[filename_col]  # e.g. "records100/00000/00001_lr"
        full_path = PTBXL_ROOT / rel_path

        print(f"[{idx+1}/{len(df)}] Processing ECG_ID={ecg_id} from {rel_path}")

        try:
            signal = preprocess_single_record(
                ecg_path=full_path,
                bandpass=True,
                fs_target=TARGET_FS,
                duration_sec=TARGET_DURATION_SEC,
            )
        except Exception as e:
            print(f"❌ Failed on ECG_ID={ecg_id}: {e}")
            continue

        out_file = OUTPUT_DIR / f"{ecg_id}.npy"
        np.save(out_file, signal)

    print("✅ PTB-XL preprocessing complete.")
    print(f"Saved preprocessed signals to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()