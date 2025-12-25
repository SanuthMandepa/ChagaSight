"""
Build structured 2D ECG images (3 × 24 × 2048) from official WFDB files.

Pipeline:
1. Load WFDB signal
2. Dataset-specific preprocessing (baseline, depadding)
3. Resample to 500 Hz + 100 Hz
4. Z-score normalization + clip [-3, 3] for 500 Hz
5. RA/LA/LL contour embedding (Kim et al.) from 500 Hz

Saves:
- 2D images: data/processed/2d_images/<dataset>/<id>_img.npy
- 100 Hz raw: data/processed/1d_signals_100hz/<dataset>/<id>.npy
- Metadata CSV: data/processed/metadata/<dataset>_metadata.csv

Usage:
python -m scripts.build_2d_images_wfdb --dataset ptbxl
python -m scripts.build_2d_images_wfdb --dataset code15 --subset 0.1
"""

import os
import argparse
import numpy as np
import wfdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import random
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

# Configuration
ALL_DATASETS = ["ptbxl", "sami_trop", "code15"]
WFDB_BASE = Path("data/official_wfdb")
IMG_BASE = Path("data/processed/2d_images")
FM_BASE = Path("data/processed/1d_signals_100hz")
METADATA_BASE = Path("data/processed/metadata")
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)
N_JOBS = max(1, os.cpu_count() - 1)

TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES_500 = int(TARGET_DURATION_SEC * 500)
TARGET_SAMPLES_100 = int(TARGET_DURATION_SEC * 100)

# Dataset-specific configs
BASELINE_CONFIG = {
    "ptbxl": {"method": "bandpass", "low_cut_hz": 0.5, "high_cut_hz": 45.0, "order": 4},
    "sami_trop": {"method": "moving_average", "window_seconds": 0.2},
    "code15": {"method": None}
}
DEPADDING_DATASETS = ["sami_trop", "code15"]

def find_wfdb_records(base_dir: Path, dataset: str) -> list[Path]:
    """Find WFDB records. For PTB-XL: only _hr files."""
    paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.hea'):
                path = Path(root) / f.replace('.hea', '')
                if dataset == "ptbxl" and not path.name.endswith('_hr'):
                    continue
                paths.append(path)
    return paths

def remove_zero_padding(signal, threshold=1e-6):
    """Remove leading/trailing zero-padding."""
    non_zero = np.any(np.abs(signal) > threshold, axis=1)
    start = np.argmax(non_zero)
    end = len(non_zero) - np.argmax(non_zero[::-1])
    return signal[start:end] if start < end else signal

def extract_label_from_header(rec_path: Path, ds: str) -> float:
    """Extract Chagas label from .hea comments, with dataset-specific mapping."""
    header = wfdb.rdheader(str(rec_path))
    label_line = [l for l in header.comments if 'Chagas label:' in l]
    if not label_line:
        raise ValueError(f"No Chagas label found in {rec_path}")
    
    val = label_line[0].split(':')[1].strip().lower()
    is_positive = val in ['true', '1', '1.0']
    
    if ds == "code15":
        return 0.8 if is_positive else 0.2
    else:
        return 1.0 if is_positive else 0.0

def process_record(args):
    ds, rec_path = args
    try:
        rec_id = rec_path.name
        signal, fields = wfdb.rdsamp(str(rec_path))
        fs = fields['fs']
        signal = signal.astype(np.float32)

        if signal.shape[1] != 12:
            raise ValueError(f"Invalid leads: {signal.shape[1]}")

        # Depadding if applicable
        if ds in DEPADDING_DATASETS:
            signal = remove_zero_padding(signal)

        # Baseline removal if applicable
        config = BASELINE_CONFIG[ds]
        if config["method"]:
            signal = remove_baseline(signal, fs=fs, **config)

        # 500 Hz path
        signal_500, _ = resample_ecg(signal, fs_in=fs, fs_out=500)
        signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_500)
        signal_500 = normalize_dataset(signal_500)
        signal_500 = np.clip(signal_500, CLIP_RANGE[0], CLIP_RANGE[1])

        # Generate image
        img = ecg_to_contour_image(signal_500, target_width=TARGET_WIDTH, clip_range=CLIP_RANGE)

        # Save image
        img_dir = IMG_BASE / ds
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / f"{rec_id}_img.npy"
        np.save(img_path, img.astype(np.float32))

        # 100 Hz path
        signal_100, _ = resample_ecg(signal, fs_in=fs, fs_out=100)
        signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_100)

        # Save 100 Hz signal
        fm_dir = FM_BASE / ds
        fm_dir.mkdir(parents=True, exist_ok=True)
        fm_path = fm_dir / f"{rec_id}.npy"
        np.save(fm_path, signal_100.astype(np.float32))

        # Extract label
        label = extract_label_from_header(rec_path, ds)

        return {
            "id": rec_id,
            "dataset": ds,
            "label": label,
            "img_path": str(img_path),
            "fm_path": str(fm_path),
            "error": None
        }

    except Exception as e:
        return {"id": rec_path.name, "dataset": ds, "error": str(e)}

def build_for_dataset(ds: str, subset: float):
    wfdb_dir = WFDB_BASE / ds
    if not wfdb_dir.exists():
        print(f"Warning: {wfdb_dir} not found — skipping {ds}")
        return 0, 0

    records = find_wfdb_records(wfdb_dir, ds)
    if subset < 1.0:
        records = random.sample(records, int(len(records) * subset))

    print(f"\nProcessing {ds.upper()}: {len(records)} records")

    args_list = [(ds, p) for p in records]

    with Pool(N_JOBS) as p:
        results = list(tqdm(p.imap(process_record, args_list), total=len(records)))

    successful = [r for r in results if r.get('error') is None]
    failed = [r for r in results if r.get('error') is not None]

    if successful:
        df = pd.DataFrame(successful)
        csv_path = METADATA_BASE / f"{ds}_metadata.csv"
        METADATA_BASE.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Metadata saved: {csv_path}")

    print(f"{ds.upper()}: Processed {len(successful)} | Failed {len(failed)}")

    if failed and len(failed) <= 10:
        print("Failed records:")
        for f in failed:
            print(f"  {f['id']}: {f['error']}")

    return len(successful), len(failed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction to process")
    parser.add_argument("--dataset", type=str, default=None, choices=ALL_DATASETS + [None],
                        help="Process only one dataset")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    total_success = 0
    total_failed = 0
    for ds in datasets:
        success, failed = build_for_dataset(ds, args.subset)
        total_success += success
        total_failed += failed

    print(f"\nTotal Processed: {total_success} | Failed: {total_failed}")

if __name__ == "__main__":
    main()