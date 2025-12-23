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
- Threading for speed (avoids multiprocessing crashes).
- Stratified subset by label.
- Robust depadding to avoid index errors.
- Robust shape handling to force (T, 12) before depadding.
- Error handling for failed records.
"""

import argparse
import time
import warnings
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.soft_labels import is_confident_label

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_WFDB = Path("data/official_wfdb")
OUTPUT_DIR_500 = Path("data/processed/1d_signals_500hz/code15")
OUTPUT_DIR_100 = Path("data/processed/1d_signals_100hz/code15")
METADATA_OUT = Path("data/processed/metadata/code15_metadata.csv")
TARGET_DURATION_SEC = 10.0
TARGET_FS_IMAGE = 500.0
TARGET_FS_FM = 100.0
PAD_MODE = 'constant'
DEPAD_THRESHOLD = 1e-6
N_JOBS = 4

def remove_zero_padding(signal, threshold=1e-6):
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    if not np.any(non_zero):
        return signal  # All zero — keep original
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end] if end > start else signal

def process_single_record(hea_path):
    try:
        rec_path = hea_path.with_suffix('')
        exam_id = hea_path.stem

        # Load signal and header
        record = wfdb.rdrecord(str(rec_path))
        signal = record.p_signal.astype(np.float32)

        # Corrected shape handling
        if signal.ndim == 3:
            signal = np.squeeze(signal)  # Squeeze all dims
        if signal.ndim != 2:
            raise ValueError(f"Unexpected signal shape after squeeze: {signal.shape}")

        # Force time-major (T, 12)
        if signal.shape[0] == 12:
            signal = signal.T  # Transpose from (12, T) to (T, 12)
        if signal.shape[1] != 12:
            raise ValueError(f"Invalid lead count: {signal.shape[1]}")

        fs_in = record.fs
        comments = record.comments

        # Parse label from comments (soft 0.2/0.8)
        label = 0.5  # Default uncertain
        for comment in comments:
            if 'Chagas label:' in comment:
                val = comment.split(':')[1].strip().lower()
                label = 0.2 if val in ['false', '0', '0.0'] else 0.8 if val in ['true', '1', '1.0'] else label
                break

        # Parse age/sex/source
        age = sex = source = np.nan
        for comment in comments:
            if 'Age:' in comment:
                val = comment.split(':')[1].strip()
                age = pd.to_numeric(val, errors='coerce')
            elif 'Sex:' in comment:
                val = comment.split(':')[1].strip().lower()
                sex = 1 if val == 'male' else 0 if val == 'female' else np.nan
            elif 'Source:' in comment:
                source = comment.split(':')[1].strip()

        # Handle NaN/inf
        signal = np.nan_to_num(signal)

        # Depad robustly (avoid index errors)
        depadded = remove_zero_padding(signal)

        # No baseline (pre-filtered)

        # Resample
        signal_500, _ = resample_ecg(signal, fs_in, TARGET_FS_IMAGE)
        signal_500 = pad_or_trim(signal_500, int(TARGET_DURATION_SEC * TARGET_FS_IMAGE), pad_mode=PAD_MODE)
        signal_100, _ = resample_ecg(signal, fs_in, TARGET_FS_FM)
        signal_100 = pad_or_trim(signal_100, int(TARGET_DURATION_SEC * TARGET_FS_FM), pad_mode=PAD_MODE)

        # Normalize 500 Hz
        signal_500 = zscore_per_lead(signal_500)

        # Save
        np.save(OUTPUT_DIR_500 / f"{exam_id}.npy", signal_500)
        np.save(OUTPUT_DIR_100 / f"{exam_id}.npy", signal_100)

        confident = is_confident_label(label, 'code15')
        return {
            'exam_id': exam_id,
            'age': age,
            'sex': sex,
            'source': source,
            'label': label,
            'confident': confident
        }

    except Exception as e:
        return {'exam_id': exam_id, 'error': str(e)}

def main(subset):
    start_time = time.time()

    # Check WFDB
    if not BASE_WFDB.exists():
        raise FileNotFoundError("Run prepare_code15_data.py first.")

    # Get all .hea
    hea_files = list((BASE_WFDB / 'code15').rglob('*.hea'))

    # Stratified subset
    if subset < 1.0:
        labels = []
        for hea in tqdm(hea_files, desc="Scanning labels for stratification"):
            comments = wfdb.rdheader(str(hea.with_suffix(''))).comments
            val = next((comment.split(':')[1].strip().lower() for comment in comments if 'Chagas label' in comment), '0.5')
            label_bin = 0 if val in ['false', '0', '0.0'] else 1
            labels.append(label_bin)
        df_hea = pd.DataFrame({'hea': hea_files, 'bin_label': labels})
        subset_df = df_hea.groupby('bin_label').sample(frac=subset, random_state=42)
        hea_files = subset_df['hea'].tolist()

    # Create dirs
    OUTPUT_DIR_500.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_100.mkdir(parents=True, exist_ok=True)
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Threading parallel
    results = Parallel(n_jobs=N_JOBS, backend='threading')(
        delayed(process_single_record)(hea) for hea in tqdm(hea_files)
    )

    # Separate
    processed = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    # Save metadata
    pd.DataFrame(processed).to_csv(METADATA_OUT, index=False)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ CODE-15% Preprocessing Complete!")
    print(f"   Total Processed: {len(processed)}")
    print(f"   Total Failed   : {len(failed)}")
    print(f"   Time Elapsed   : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"   Saved to      :")
    print(f"       Images (500 Hz) → {OUTPUT_DIR_500}")
    print(f"       FM (100 Hz) → {OUTPUT_DIR_100}")
    print("=" * 60)

    if failed:
        print("\n❌ First 10 failed records:")
        for r in failed[:10]:
            print(f"   {r['exam_id']}: {r['error']}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")

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