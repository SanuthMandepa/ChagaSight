# scripts/preprocess_code15.py
"""
Preprocess CODE-15% from official WFDB folder (after prepare_code15_data.py).
Generates 500 Hz z-scored and 100 Hz raw .npy.
Adds soft labels to metadata.

Dataset reference: Zenodo CODE-15% (Ribeiro et al., 2021).

Usage:
python scripts/preprocess_code15.py --subset 1.0

Notes:
- Scans ALL .hea files in data/official_wfdb/code15/
- No baseline (already filtered per paper).
- Depadding for short/padded signals.
- Soft labels: 0.2/0.8 from WFDB header.
- Multiprocessing for speed.
- Warnings suppressed specific.
- Error handling + logging.
"""

import os
from pathlib import Path
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import wfdb

from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import get_chagas_label

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
RAW_DIR = Path("data/official_wfdb/code15")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/code15")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/code15")
METADATA_OUTPUT = Path("data/processed/metadata/code15_processed.csv")

TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000

N_JOBS = cpu_count() - 1

DEPAD_THRESHOLD = 1e-6  # Configurable

def remove_zero_padding(signal, threshold=DEPAD_THRESHOLD):
    """Remove leading/trailing zero-padding."""
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end]

def preprocess_single_record(rec_path: Path):
    """Preprocess one WFDB record."""
    try:
        exam_id = rec_path.name
        record = wfdb.rdrecord(str(rec_path))
        signal = record.p_signal.astype(np.float32)
        raw_fs = record.fs

        # Check 12 leads
        if signal.shape[1] != 12:
            raise ValueError("Not 12 leads")

        # Depad
        depadded = remove_zero_padding(signal)

        # No baseline (already filtered)
        filtered = depadded

        # 500 Hz + z-score for images
        signal_500, _ = resample_ecg(filtered, raw_fs, TARGET_FS_IMAGE)
        signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
        normalized_500 = zscore_per_lead(signal_500)

        # 100 Hz raw for FM
        signal_100, _ = resample_ecg(filtered, raw_fs, TARGET_FS_FM)
        signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

        # Save
        np.save(OUTPUT_DIR_500 / f"{exam_id}.npy", normalized_500)
        np.save(OUTPUT_DIR_100 / f"{exam_id}.npy", signal_100)

        # Soft label from header
        chagas_comment = [c for c in record.comments if 'Chagas label' in c]
        chagas = True if chagas_comment and 'True' in chagas_comment[0] else False
        label = get_chagas_label({'chagas': chagas}, 'code15')

        return {
            "exam_id": exam_id,
            "path_500hz": str(OUTPUT_DIR_500 / f"{exam_id}.npy"),
            "path_100hz": str(OUTPUT_DIR_100 / f"{exam_id}.npy"),
            "original_length": signal.shape[0],
            "depadded_length": depadded.shape[0],
            "label": label
        }

    except Exception as e:
        return {"exam_id": exam_id, "error": str(e)}

def main(subset_fraction: float = 1.0):
    start_time = time.time()

    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Collect ALL .hea files recursively
    record_paths = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith('.hea'):
                rec_path = Path(root) / f.replace('.hea', '')
                record_paths.append(rec_path)

    print(f"📂 Found {len(record_paths)} valid WFDB records")

    # Subset if needed
    if subset_fraction < 1.0:
        record_paths = record_paths[:int(len(record_paths) * subset_fraction)]
        print(f"⚠️ SUBSET MODE: Using {len(record_paths)} records ({subset_fraction*100:.1f}% of total)")

    args_list = record_paths

    # Multiprocess
    with Pool(N_JOBS) as p:
        results = list(tqdm(p.imap(preprocess_single_record, args_list), total=len(args_list), desc="Processing CODE-15%"))

    # Separate processed/failed
    processed_records = [r for r in results if 'error' not in r]
    failed_records = [r for r in results if 'error' in r]

    # Save metadata
    if processed_records:
        pd.DataFrame(processed_records).to_csv(METADATA_OUTPUT, index=False)
        print(f"\n💾 Metadata saved → {METADATA_OUTPUT}")

    # Summary
    elapsed = time.time() - start_time
    total_processed = len(processed_records)
    total_failed = len(failed_records)

    print("\n" + "=" * 60)
    print("✅ CODE-15% preprocessing complete!")
    print("=" * 60)
    print(f"   Processed : {total_processed:,}")
    print(f"   Failed    : {total_failed}")
    print(f"   Total time: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"   Saved to  :")
    print(f"       Images (500 Hz) → {OUTPUT_DIR_500}")
    print(f"       FM (100 Hz) → {OUTPUT_DIR_100}")
    print("=" * 60)

    if failed_records:
        print("\n❌ First 10 failed records:")
        for r in failed_records[:10]:
            print(f"   {r['exam_id']}: {r['error']}")
        if len(failed_records) > 10:
            print(f"   ... and {len(failed_records) - 10} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CODE-15% from WFDB")
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of dataset to process (default=1.0 → full)"
    )
    args = parser.parse_args()
    main(args.subset)