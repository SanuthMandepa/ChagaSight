"""
Preprocess PTB-XL into standardized 1D numpy arrays.

Output:
    data/processed/1d_signals/ptbxl/<ecg_id>.npy
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim


PTBXL_ROOT = Path("data/raw/ptbxl")
OUTPUT_DIR = Path("data/processed/1d_signals/ptbxl")

TARGET_FS = 400.0
TARGET_DURATION_SEC = 10.0


def preprocess_single_record(ecg_path: Path, fs_target: float, duration_sec: float):
    """Load → baseline-remove → resample → pad/trim → return numpy array."""
    record = wfdb.rdrecord(str(ecg_path))
    signal = record.p_signal.astype(np.float32)
    fs_in = float(record.fs)

    # 1) Baseline removal
    signal = remove_baseline(signal, fs=fs_in, method="highpass")

    # 2) Resample
    signal, fs_rs = resample_ecg(signal, fs_in=fs_in, fs_out=fs_target)

    # 3) Pad or trim
    target_len = int(round(duration_sec * fs_rs))
    signal = pad_or_trim(signal, target_length=target_len)

    return signal


def main():
    print("Loading PTB-XL metadata...")
    df = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        ecg_id = int(row["ecg_id"])
        rel_path = row["filename_hr"]  # ex: "records500/00000/00001_hr"

        full_path = PTBXL_ROOT / rel_path

        print(f"Processing ECG {ecg_id} at {full_path}")

        try:
            signal = preprocess_single_record(
                ecg_path=full_path,
                fs_target=TARGET_FS,
                duration_sec=TARGET_DURATION_SEC,
            )
        except Exception as e:
            print(f"❌ Failed on {ecg_id}: {e}")
            continue

        np.save(OUTPUT_DIR / f"{ecg_id}.npy", signal)

    print("✅ PTB-XL preprocessing complete.")


if __name__ == "__main__":
    main()
