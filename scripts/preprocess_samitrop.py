# scripts/preprocess_samitrop.py
"""
Preprocess SaMi-Trop from WFDB (after prepare_samitrop_data.py).
Generates 500 Hz z-scored and 100 Hz raw .npy.
Adds soft labels to metadata.

Dataset reference: Zenodo SaMi-Trop (Ribeiro et al., 2021).

Usage:
python scripts/preprocess_samitrop.py

Notes:
- Loads from WFDB for Challenge alignment.
- Depadding for 4096 → 4000/2800 (per paper).
- Moving-average baseline.
- Soft labels: 1.0 (positives).
- Warnings suppressed.
- Error handling for failed records.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import get_chagas_label

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Configuration constants
RAW_DIR = Path("data/official_wfdb/sami_trop")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/sami_trop")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/sami_trop")
METADATA_OUTPUT = Path("data/processed/metadata/sami_trop_processed.csv")

TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_SEC * TARGET_FS_IMAGE)  # 5000
TARGET_SAMPLES_FM = int(TARGET_DURATION_SEC * TARGET_FS_FM)        # 1000

ORIGINAL_FS = 400.0  # Native

# Baseline removal config for SaMi-Trop (moving-average per paper)
BASELINE_METHOD = "moving_average"
BASELINE_KWARGS = {"window_seconds": 0.2}

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

def preprocess_single_record(record_path: Path, original_fs: float) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Preprocess a single WFDB record:
    1. Load signal.
    2. Depad zeros.
    3. Apply baseline removal.
    4. Resample to 500 Hz (z-scored) and 100 Hz (raw).
    5. Pad/trim to fixed length.

    Parameters:
    - record_path: Path to WFDB record (without extension).
    - original_fs: Sampling frequency.

    Returns:
    - signal_500: Z-scored at 500 Hz for images.
    - signal_100: Raw at 100 Hz for FM.
    - original_length: For metadata.
    """
    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception as e:
        raise IOError(f"Failed to read WFDB record {record_path}: {e}")

    signal = record.p_signal.astype(np.float32)
    if signal.shape[1] != 12:
        warnings.warn(f"Record {record_path} has {signal.shape[1]} leads, forcing to 12")
        if signal.shape[1] < 12:
            padded = np.zeros((signal.shape[0], 12), dtype=np.float32)
            padded[:, :signal.shape[1]] = signal
            signal = padded
        else:
            signal = signal[:, :12]

    original_length = signal.shape[0]  # Save here, before depad

    # Depad
    depadded = remove_zero_padding(signal)

    # Baseline removal
    filtered = remove_baseline(depadded, fs=original_fs, method=BASELINE_METHOD, **BASELINE_KWARGS)

    # 500 Hz + z-score for images
    signal_500, _ = resample_ecg(filtered, original_fs, TARGET_FS_IMAGE)
    signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_IMAGE)
    signal_500 = zscore_per_lead(signal_500)

    # 100 Hz raw for FM
    signal_100, _ = resample_ecg(filtered, original_fs, TARGET_FS_FM)
    signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_FM)

    return signal_500.astype(np.float32), signal_100.astype(np.float32), original_length

def main() -> None:
    start_time = time.time()

    # Create directories
    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Find all WFDB records
    hea_files = list(RAW_DIR.glob("*.hea"))
    print(f"📄 Found {len(hea_files)} WFDB records")

    processed_records = []
    failed_records = []

    for hea_path in tqdm(hea_files, desc="Processing SaMi-Trop"):
        exam_id = int(hea_path.stem)  # WFDB name = exam_id

        try:
            signal_500, signal_100, original_length = preprocess_single_record(hea_path.with_suffix(''), ORIGINAL_FS)

            # Save
            np.save(OUTPUT_DIR_500 / f"{exam_id}.npy", signal_500)
            np.save(OUTPUT_DIR_100 / f"{exam_id}.npy", signal_100)

            # Metadata with soft label
            label = get_chagas_label({'exam_id': exam_id}, 'sami_trop')  # 1.0 for SaMi-Trop
            processed_records.append({
                "exam_id": exam_id,
                "path_500hz": str(OUTPUT_DIR_500 / f"{exam_id}.npy"),
                "path_100hz": str(OUTPUT_DIR_100 / f"{exam_id}.npy"),
                "original_length": original_length,
                "label": label
            })

        except Exception as e:
            failed_records.append({"exam_id": exam_id, "error": str(e)})

    # Save metadata
    if processed_records:
        pd.DataFrame(processed_records).to_csv(METADATA_OUTPUT, index=False)
        print(f"\n💾 Metadata saved → {METADATA_OUTPUT}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ SaMi-Trop preprocessing complete!")
    print("=" * 60)
    print(f"   Processed : {len(processed_records)}")
    print(f"   Failed    : {len(failed_records)}")
    print(f"   Total time: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
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
    main()