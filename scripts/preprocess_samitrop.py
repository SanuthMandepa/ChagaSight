"""
Preprocess CODE-15% dataset into cleaned 1D ECG signals.

Pipeline (aligned with PTB-XL & SaMi-Trop):
    - baseline removal (high-pass)
    - resampling to 400 Hz
    - padding/truncation to 10 seconds
    - per-lead z-score normalization

Inputs (expected layout):
    data/raw/code15/
        exams.csv
        exams_part0.hdf5
        exams_part1.hdf5
        ...
        exams_part17.hdf5

Outputs:
    data/processed/1d_signals/code15/{exam_id}.npy
        shape = (4000, 12)  # 10s @ 400 Hz
"""

import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
RAW_DIR = Path("data/raw/code15")
OUT_DIR = Path("data/processed/1d_signals/code15")

# Use SAME fs and duration as PTB-XL & SaMi-Trop
TARGET_FS = 400.0                 # Hz
TARGET_DURATION_SEC = 10.0        # seconds
TARGET_SAMPLES = int(TARGET_FS * TARGET_DURATION_SEC)  # 4000


def open_h5(path: Path):
    """Open an HDF5 file."""
    return h5py.File(path, "r")


def get_tracings_and_ids(h5):
    """
    Robustly get tracings and exam IDs from CODE-15 HDF5.

    Handles both:
        - 'tracings' / 'exam_id'
        - 'signal' / 'id_exam'
    """
    # tracings / signal
    if "tracings" in h5:
        tracings = h5["tracings"]
    elif "signal" in h5:
        tracings = h5["signal"]
    else:
        raise KeyError("HDF5 file missing 'tracings' or 'signal' dataset.")

    # exam_id / id_exam
    if "exam_id" in h5:
        exam_ids = np.array(h5["exam_id"])
    elif "id_exam" in h5:
        exam_ids = np.array(h5["id_exam"])
    else:
        raise KeyError("HDF5 file missing 'exam_id' or 'id_exam' dataset.")

    return tracings, exam_ids


def preprocess_single_ecg(ecg_4096x12: np.ndarray,
                          fs_in: float = 400.0,
                          fs_out: float = TARGET_FS,
                          duration_sec: float = TARGET_DURATION_SEC) -> np.ndarray:
    """
    Apply the unified 1D preprocessing pipeline to one ECG:

        - baseline removal (high-pass)
        - resampling to fs_out
        - pad/trim to duration_sec
        - z-score normalization

    Returns:
        np.ndarray of shape (target_length, 12)
    """
    # 1) Baseline removal (same style as PTB-XL)
    cleaned = remove_baseline(
        ecg_4096x12,
        fs=fs_in,
        method="highpass",
        highpass_cutoff=0.5,  # Hz
        filter_order=5
    )

    # 2) Resample (will keep fs_out == TARGET_FS)
    ecg_rs, fs_rs = resample_ecg(cleaned, fs_in=fs_in, fs_out=fs_out)

    # 3) Pad/trim to exactly duration_sec
    target_length = int(duration_sec * fs_rs)
    ecg_fixed = pad_or_trim(ecg_rs, target_length=target_length)

    # 4) Per-lead z-score normalization
    ecg_norm = zscore_per_lead(ecg_fixed)

    return ecg_norm


def main():
    print("⏱️ Starting: CODE-15% 1D preprocessing")

    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load CSV metadata ----
    csv_path = RAW_DIR / "exams.csv"
    df = pd.read_csv(csv_path)

    print(f"📄 Loaded metadata: {csv_path}")
    print(f"🔢 Total exams in CSV: {len(df)}")

    # Group by trace_file (exams_part*.hdf5)
    part_groups = df.groupby("trace_file")

    total_processed = 0

    # ---- Loop over each HDF5 part ----
    for trace_file, group in part_groups:
        h5_path = RAW_DIR / trace_file
        print(f"\n📂 Opening HDF5: {trace_file} (rows in CSV group: {len(group)})")

        if not h5_path.exists():
            print(f"❌ HDF5 file not found: {h5_path}, skipping this part.")
            continue

        with open_h5(h5_path) as h5:
            tracings, exam_ids = get_tracings_and_ids(h5)

            # Build lookup: exam_id → index in this HDF5
            index_map = {int(exam_ids[i]): i for i in range(len(exam_ids))}

            # Process each row belonging to this HDF5
            for i, (_, row) in enumerate(group.iterrows(), start=1):
                exam_id = int(row["exam_id"])

                if exam_id not in index_map:
                    print(f"⚠️ exam_id {exam_id} not found in {trace_file}, skipping.")
                    continue

                idx = index_map[exam_id]
                raw = np.array(tracings[idx])  # expected shape: (4096, 12)

                # Sanity check
                if raw.ndim != 2 or raw.shape[1] != 12:
                    print(f"⚠️ Unexpected shape for exam_id {exam_id}: {raw.shape}, skipping.")
                    continue

                # Preprocess 1D ECG
                try:
                    signal = preprocess_single_ecg(
                        raw,
                        fs_in=400.0,
                        fs_out=TARGET_FS,
                        duration_sec=TARGET_DURATION_SEC,
                    )
                except Exception as e:
                    print(f"❌ Failed on exam_id {exam_id}: {e}")
                    continue

                # Save as {exam_id}.npy (no label files here)
                out_npy = OUT_DIR / f"{exam_id}.npy"
                np.save(out_npy, signal)

                total_processed += 1

                # Progress log every 100 exams (and on first)
                if i == 1 or i % 100 == 0:
                    print(
                        f"[{i}/{len(group)}] exam_id={exam_id} "
                        f"→ saved {out_npy.name} shape={signal.shape}"
                    )

    elapsed = time.time() - t0
    print(
        f"\n⏱️ CODE-15% 1D preprocessing completed in "
        f"{elapsed:.2f} s ({elapsed/60:.2f} min)"
    )
    print(f"✅ Total processed exams: {total_processed}")
    print(f"💾 Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
