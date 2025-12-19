"""
Preprocess CODE-15% dataset into standardized 1D ECG signals.

Dataset reference (Zenodo):
- "CODE-15%: a large scale annotated dataset of 12-lead ECGs"
  Ribeiro et al., 2021, DOI: 10.5281/zenodo.4916206

Raw files (expected layout):
    data/raw/code15/
        exams.csv
        exams_part0.hdf5 ... exams_part17.hdf5  (unzipped from exams_part*.zip)

HDF5 layout per shard:
    exam_id : (N,)      – exam IDs
    tracings: (N, 4096, 12) – signals at 400 Hz (some 7s padded symmetrically to 4096)

Dual preprocessing pipeline (aligned with Kim et al. 2025 and Van Santvliet et al. 2025):
    1. Load signal from correct HDF5 shard.
    2. Remove symmetric zero-padding (for 7-second recordings).
    3. NO baseline removal → signals are already filtered in the dataset.
    4. Resample:
         • To 500 Hz → for 2D RA/LA/LL image generation (Kim et al.)
         • To 100 Hz → for 1D foundation model (Van Santvliet et al.)
    5. Pad/trim to fixed 10 seconds:
         • 5000 samples at 500 Hz
         • 1000 samples at 100 Hz
    6. Per-lead z-score normalization (only for 500 Hz image path).
    7. Save as:
         data/processed/1d_signals_500hz/code15/<exam_id>.npy   # z-scored, for images
         data/processed/1d_signals_100hz/code15/<exam_id>.npy   # raw amplitudes, for FM

Features:
    - Default --subset 0.1 (10%) to avoid 50+ GB storage
    - Efficient per-shard processing
    - Progress bars and detailed logging
"""

import argparse
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
RAW_DIR = Path("data/raw/code15")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/code15")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/code15")
METADATA_OUTPUT = Path("data/processed/metadata/code15_processed.csv")

ORIGINAL_FS = 400.0
TARGET_FS_IMAGE = 500.0   # Kim et al. 2025
TARGET_FS_FM = 100.0      # Van Santvliet et al. 2025
TARGET_DURATION_SEC = 10.0

TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000


def remove_zero_padding(signal: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Remove symmetric zero-padding from CODE-15% signals."""
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end]


def main(subset_fraction: float = 0.1) -> None:
    start_time = time.time()

    # Create output directories
    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    csv_path = RAW_DIR / "exams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"exams.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"📄 Loaded metadata: {len(df)} exams")

    # Apply subset
    if subset_fraction < 1.0:
        df = df.sample(frac=subset_fraction, random_state=42).reset_index(drop=True)
        print(f"⚠️  SUBSET MODE: Using {len(df)} exams ({subset_fraction*100:.1f}% of total)")

    processed_records = []
    failed_records = []

    # Group by trace_file (shard)
    for trace_file, group in tqdm(df.groupby("trace_file"), desc="Shards"):
        h5_path = RAW_DIR / trace_file
        if not h5_path.exists():
            warnings.warn(f"HDF5 shard not found: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as h5:
            if "exam_id" not in h5 or "tracings" not in h5:
                warnings.warn(f"Missing datasets in {trace_file}")
                continue

            exam_ids_h5 = np.array(h5["exam_id"])
            tracings = h5["tracings"]
            id_to_idx = {int(exam_ids_h5[i]): i for i in range(len(exam_ids_h5))}

            for _, row in tqdm(group.iterrows(), total=len(group), desc=f"{trace_file}", leave=False):
                exam_id = int(row["exam_id"])

                if exam_id not in id_to_idx:
                    failed_records.append({"exam_id": exam_id, "error": "ID not in HDF5"})
                    continue

                idx = id_to_idx[exam_id]
                raw_signal = np.asarray(tracings[idx]).astype(np.float32)  # (4096, 12)

                try:
                    # 1. Remove zero-padding
                    depadded = remove_zero_padding(raw_signal)

                    # NO baseline removal – dataset is already filtered

                    # 2. Image path: 500 Hz + z-score
                    signal_500, _ = resample_ecg(depadded, ORIGINAL_FS, TARGET_FS_IMAGE)
                    signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
                    normalized_500 = zscore_per_lead(signal_500)

                    # 3. FM path: 100 Hz, raw
                    signal_100, _ = resample_ecg(depadded, ORIGINAL_FS, TARGET_FS_FM)
                    signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

                    # Save
                    np.save(OUTPUT_DIR_500 / f"{exam_id}.npy", normalized_500)
                    np.save(OUTPUT_DIR_100 / f"{exam_id}.npy", signal_100)

                    processed_records.append({
                        "exam_id": exam_id,
                        "trace_file": trace_file,
                        "path_500hz": str(OUTPUT_DIR_500 / f"{exam_id}.npy"),
                        "path_100hz": str(OUTPUT_DIR_100 / f"{exam_id}.npy"),
                        "original_length": raw_signal.shape[0],
                        "depadded_length": depadded.shape[0],
                    })

                except Exception as e:
                    warnings.warn(f"Failed exam_id {exam_id}: {e}")
                    failed_records.append({"exam_id": exam_id, "error": str(e)})

    # Save metadata
    if processed_records:
        pd.DataFrame(processed_records).to_csv(METADATA_OUTPUT, index=False)
        print(f"\n💾 Metadata saved → {METADATA_OUTPUT}")

    # Summary
    elapsed = time.time() - start_time
    total_processed = len(processed_records)
    total_failed = len(failed_records)

    print("\n" + "=" * 60)
    print("🎉 CODE-15% preprocessing complete!")
    print("=" * 60)
    print(f"   Processed : {total_processed}")
    print(f"   Failed    : {total_failed}")
    print(f"   Total time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"   Saved to  :")
    print(f"       Images (500 Hz) → {OUTPUT_DIR_500}")
    print(f"       FM     (100 Hz) → {OUTPUT_DIR_100}")
    print("=" * 60)

    if failed_records:
        print("\n❌ First 10 failed records:")
        for r in failed_records[:10]:
            print(f"   {r['exam_id']}: {r['error']}")
        if len(failed_records) > 10:
            print(f"   ... and {len(failed_records) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=float,
        default=0.1,
        help="Fraction of dataset to process (default=0.1 → 10%)"
    )
    args = parser.parse_args()
    main(args.subset)