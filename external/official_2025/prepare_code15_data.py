# scripts/preprocess_code15.py
"""
Preprocess CODE-15% from WFDB (after prepare_code15_data.py).
Generates 500 Hz z-scored and 100 Hz raw .npy.
Adds soft labels to metadata.

Dataset reference: Zenodo CODE-15% (Ribeiro et al., 2021).

Usage:
python scripts/preprocess_code15.py --subset 0.1

Notes:
- Loads from WFDB for Challenge alignment.
- Depadding for 4096 → 4000/2800 (per paper).
- No baseline (already filtered per paper).
- Soft labels: 0.2/0.8 for negatives/positives.
- Multiprocessing for speed on large subsets.
- Warnings suppressed.
- Error handling for failed records.
"""
import argparse
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import wfdb

from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import get_chagas_label

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Configuration constants
RAW_DIR = Path("data/official_wfdb/code15")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/code15")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/code15")
METADATA_OUTPUT = Path("data/processed/metadata/code15_processed.csv")

TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000

ORIGINAL_FS = 400.0  # Native

# No baseline for CODE-15% (already filtered per paper)
BASELINE_METHOD = None

N_JOBS = cpu_count() - 1

def remove_zero_padding(signal, threshold=1e-6):
    """Remove leading/trailing zero-padding.
    
    Parameters:
    - signal: ECG array (T, 12)
    - threshold: Amplitude below which is considered zero.
    
    Returns:
    - Depadded signal.
    """
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end]

def preprocess_single_record(args) -> dict:
    """Preprocess single CODE-15% WFDB record (for multiprocessing).
    
    Parameters:
    - args: Tuple (exam_id, wfdb_path)
    
    Returns:
    - Dict with paths, lengths, label, or error.
    """
    exam_id, wfdb_path = args

    try:
        record = wfdb.rdrecord(str(wfdb_path))
        signal = record.p_signal.astype(np.float32)
        raw_fs = record.fs

        if signal.shape[1] != 12:
            warnings.warn(f"Record {wfdb_path} has {signal.shape[1]} leads, forcing to 12")
            if signal.shape[1] < 12:
                padded = np.zeros((signal.shape[0], 12), dtype=np.float32)
                padded[:, :signal.shape[1]] = signal
                signal = padded
            else:
                signal = signal[:, :12]

        original_length = signal.shape[0]  # Save here

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

        # Metadata with soft label (parse from header)
        chagas_comment = [c for c in record.comments if 'Chagas label' in c]
        chagas = True if chagas_comment and 'True' in chagas_comment[0] else False
        label = get_chagas_label({'chagas': chagas}, 'code15')

        return {
            "exam_id": exam_id,
            "path_500hz": str(OUTPUT_DIR_500 / f"{exam_id}.npy"),
            "path_100hz": str(OUTPUT_DIR_100 / f"{exam_id}.npy"),
            "original_length": original_length,
            "depadded_length": depadded.shape[0],
            "label": label
        }

    except Exception as e:
        return {"exam_id": exam_id, "error": str(e)}

def main(subset_fraction: float = 0.1):
    start_time = time.time()

    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata to get exam_ids (for subset)
    csv_path = Path("data/raw/code15/exams.csv")
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded metadata: {len(df)} exams")

    # Apply subset
    if subset_fraction < 1.0:
        df = df.sample(frac=subset_fraction, random_state=42).reset_index(drop=True)
        print(f"⚠️ SUBSET MODE: Using {len(df)} exams ({subset_fraction*100:.1f}% of total)")

    # Prepare WFDB paths
    args_list = []
    for _, row in df.iterrows():
        exam_id = int(row["exam_id"])
        wfdb_path = RAW_DIR / f"{exam_id}"
        if wfdb_path.with_suffix('.hea').exists():
            args_list.append((exam_id, wfdb_path))
        else:
            print(f"Warning: WFDB missing for {exam_id}")

    print(f"📂 Found {len(args_list)} valid WFDB records")

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
    print(f"   Processed : {total_processed}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=float,
        default=0.1,
        help="Fraction of dataset to process (default=0.1 → 10%)"
    )
    args = parser.parse_args()
    main(args.subset)