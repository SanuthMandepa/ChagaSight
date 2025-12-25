# scripts/preprocess_ptbxl.py
"""
Preprocess PTB-XL into standardized 1D numpy arrays.

Dataset reference (PhysioNet):
- PTB-XL, a large publicly available electrocardiography dataset
  Wagner et al., 2020, DOI: 10.13026/kfzx-aw45 (version 1.0.3)

Raw files (expected layout):
    data/raw/ptbxl/
        ptbxl_database.csv         # Metadata: ecg_id, filename_hr/lr, age, sex, etc.
        scp_statements.csv         # Diagnostic codes (not used here)
        records100/                # Low-res 100 Hz WFDB files (not used)
        records500/                # High-res 500 Hz WFDB files (preferred for better resolution)

Dual preprocessing pipeline (aligned with Kim et al. 2025 and Van Santvliet et al. 2025):
    1. Load WFDB record (prefer high-res 500 Hz for better signal quality).
    2. Apply bandpass filtering (0.5–45 Hz) for baseline wander and noise removal (dataset-specific).
    3. Resample:
         • To 500 Hz → for 2D RA/LA/LL image generation (Kim et al.) – preserves details for embedding.
         • To 100 Hz → for 1D foundation model (Van Santvliet et al.) – raw amplitudes for pretraining.
    4. Pad/trim to fixed 10 seconds:
         • 5000 samples at 500 Hz (padded/trimmed symmetrically).
         • 1000 samples at 100 Hz.
    5. Per-lead z-score normalization (only for 500 Hz image path – mean 0, std 1 per lead).
    6. Save as:
         data/processed/1d_signals_500hz/ptbxl/<ecg_id>.npy   # z-scored, for images (Kim et al.)
         data/processed/1d_signals_100hz/ptbxl/<ecg_id>.npy   # raw amplitudes, for FM (Van Santvliet et al.)

This script generates preprocessed signals used by:
    - scripts/build_images.py                  → 2D contour images (from 500 Hz)
    - Future FM pretraining / fine-tuning      → (from 100 Hz)
    - Hybrid alignment experiments

Usage:
python scripts/preprocess_ptbxl.py

Notes:
- Uses high-res 500 Hz (records500/) for better alignment with papers (sharper signals).
- Soft labels added to metadata (0.0 for PTB-XL as negatives).
- Warnings suppressed for clean output.
- Detailed error handling for failed records.
- Parallelized with multiprocessing for speed (N_JOBS adjustable).
- No depadding needed for PTB-XL (full 10s signals).
"""

import os
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import get_chagas_label

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Configuration constants
PTBXL_ROOT = Path("data/raw/ptbxl")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/ptbxl")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/ptbxl")
METADATA_OUTPUT = Path("data/processed/metadata/ptbxl_processed.csv")

TARGET_FS_IMAGE = 500.0   # For 2D images (Kim et al.)
TARGET_FS_FM = 100.0      # For 1D FM (Van Santvliet et al.)
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000

# Prefer high-res records (500 Hz) for better quality
USE_HIGH_RES = True
ORIGINAL_FS = 500.0 if USE_HIGH_RES else 100.0
FILENAME_COL = "filename_hr" if USE_HIGH_RES else "filename_lr"

# Baseline removal config (bandpass for PTB-XL per paper alignment)
BASELINE_METHOD = "bandpass"
BASELINE_KWARGS = {
    "low_cut_hz": 0.5,
    "high_cut_hz": 45.0,
    "order": 4,
}

# Parallelization config
N_JOBS = cpu_count() - 1  # Leave one core free for system

def preprocess_single_record(args) -> dict:
    """
    Preprocess a single PTB-XL record.

    Parameters:
    - args: Tuple (ecg_id, row) from metadata DataFrame.

    Returns:
    - Dict with metadata or error.
    """
    ecg_id, row = args
    try:
        rel_path = row[FILENAME_COL]
        full_path = PTBXL_ROOT / rel_path

        # Load record
        record = wfdb.rdrecord(str(full_path))
        signal = record.p_signal.astype(np.float32)
        fs = record.fs

        # No depadding for PTB-XL (full 10s)

        # Step 2: Baseline removal
        filtered = remove_baseline(signal, fs=fs, method=BASELINE_METHOD, **BASELINE_KWARGS)

        # Step 3 & 4: Image path - 500 Hz + z-score
        signal_500, _ = resample_ecg(filtered, fs_in=fs, fs_out=TARGET_FS_IMAGE)
        signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
        signal_500 = zscore_per_lead(signal_500)

        # FM path - 100 Hz, raw (no z-score)
        signal_100, _ = resample_ecg(filtered, fs_in=fs, fs_out=TARGET_FS_FM)
        signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

        # Save with numeric ID
        np.save(OUTPUT_DIR_500 / f"{ecg_id}.npy", signal_500.astype(np.float32))
        np.save(OUTPUT_DIR_100 / f"{ecg_id}.npy", signal_100.astype(np.float32))

        # Metadata with soft label
        label = get_chagas_label({'ecg_id': ecg_id}, 'ptbxl')  # 0.0 for PTB-XL
        return {
            "ecg_id": ecg_id,
            "original_path": rel_path,
            "original_fs": fs,
            "original_length": signal.shape[0],
            "path_500hz": str(OUTPUT_DIR_500 / f"{ecg_id}.npy"),
            "path_100hz": str(OUTPUT_DIR_100 / f"{ecg_id}.npy"),
            "label": label,
        }

    except Exception as e:
        return {
            "ecg_id": ecg_id,
            "error": str(e)
        }

def main() -> None:
    start_time = time.time()

    # Create directories
    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    db_path = PTBXL_ROOT / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata not found: {db_path}")

    df = pd.read_csv(db_path)
    print(f"📄 Loaded {len(df)} PTB-XL records")

    args_list = [(int(row["ecg_id"]), row) for _, row in df.iterrows() if pd.notna(row[FILENAME_COL])]

    with Pool(N_JOBS) as p:
        results = list(tqdm(p.imap(preprocess_single_record, args_list), total=len(args_list), desc="Processing PTB-XL"))

    processed_records = []
    failed_records = []

    for result in results:
        if "error" in result:
            failed_records.append(result)
        else:
            processed_records.append(result)

    # Save metadata
    if processed_records:
        pd.DataFrame(processed_records).to_csv(METADATA_OUTPUT, index=False)
        print(f"💾 Metadata saved → {METADATA_OUTPUT}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ PTB-XL preprocessing complete!")
    print("=" * 60)
    print(f"   Processed : {len(processed_records)}")
    print(f"   Failed    : {len(failed_records)}")
    print(f"   Total time: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"   Saved to  :")
    print(f"       500 Hz → {OUTPUT_DIR_500}")
    print(f"       100 Hz → {OUTPUT_DIR_100}")
    print("=" * 60)

    if failed_records:
        print("\n❌ First 10 failed records:")
        for r in failed_records[:10]:
            print(f"   {r['ecg_id']}: {r['error']}")
        if len(failed_records) > 10:
            print(f"   ... and {len(failed_records) - 10} more")

if __name__ == "__main__":
    main()