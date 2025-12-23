"""
Build structured 2D ECG images (3 × 24 × 2048) from official WFDB files.

Pipeline:
1. Load WFDB signal
2. Dataset-specific baseline removal (bandpass PTB-XL, moving-average others)
3. Resample to 500 Hz + 100 Hz (dual FS)
4. Z-score normalization for 500 Hz
5. Clip to [-3, 3] for 500 Hz
6. RA/LA/LL contour embedding (Kim et al.) from 500 Hz

Saves 100 Hz raw for FM (Van Santvliet).
Handles PTB-XL subfolders.
Parallelized for speed.

Usage:
python -m scripts.build_2d_images_wfdb --subset 0.1

Notes:
- Parallel with multiprocessing for speed.
- Subset random sample for large CODE-15%.
- Clipping warnings suppressed.
- Error handling for failed records.
- Depadding added for SaMi-Trop/CODE-15% short signals.
- Signal shape forced to (T, 12) to avoid index errors.
- Warnings suppressed for clean output.
- PTB-XL handling: No depadding (full 5000 samples at 500 Hz), check fs == 500 to skip unnecessary resample.
"""

import os
import argparse
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import sys
import random
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Fix import path for module run (-m)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg, pad_or_trim
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

# Configuration
DATASETS = ["ptbxl", "sami_trop", "code15"]  # Full run
WFDB_BASE = Path("data/official_wfdb")
IMG_BASE = Path("data/processed/2d_images")
FM_BASE = Path("data/processed/1d_signals_100hz")  # For 1D FM
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)
N_JOBS = 6  # Adjust for your i7 (6–8 cores)

# Dataset-specific configs (aligned with papers and old scripts)
BASELINE_CONFIG = {
    "ptbxl": {"method": "bandpass", "low_cut_hz": 0.5, "high_cut_hz": 45.0, "order": 4},
    "sami_trop": {"method": "moving_average", "window_seconds": 0.2},
    "code15": {"method": None}  # No baseline, already filtered
}

DEPADDING_DATASETS = ["ptbxl","sami_trop", "code15"]  # Depad for these
TARGET_DURATION_SEC = 10.0  # Standard 10s
TARGET_SAMPLES_500 = int(TARGET_DURATION_SEC * 500)
TARGET_SAMPLES_100 = int(TARGET_DURATION_SEC * 100)

def process_record(args):
    ds, rec_path = args
    try:
        signal, fields = wfdb.rdsamp(str(rec_path))
        fs = fields['fs']
        
        # Force to (T, 12) to avoid index errors
        if signal.shape[1] != 12:
            raise ValueError(f"Invalid leads: {signal.shape[1]}")

        # Depadding for SaMi-Trop/CODE-15%
        if ds in DEPADDING_DATASETS:
            non_zero = np.any(np.abs(signal) > 1e-6, axis=1)
            start = np.argmax(non_zero)
            end = len(non_zero) - np.argmax(non_zero[::-1])
            signal = signal[start:end]
        
        # Dataset-specific baseline removal
        method = BASELINE_CONFIG[ds]["method"]
        kwargs = {k: v for k, v in BASELINE_CONFIG[ds].items() if k != "method"}
        if method:
            signal = remove_baseline(signal, fs=fs, method=method, **kwargs)
        
        # Resample if needed (skip if fs == target)
        if fs != 500:
            signal_500, _ = resample_ecg(signal, fs_in=fs, fs_out=500)
        else:
            signal_500 = signal
        signal_500 = pad_or_trim(signal_500, TARGET_SAMPLES_500)
        
        signal_500 = normalize_dataset(signal_500)
        signal_500 = np.clip(signal_500, CLIP_RANGE[0], CLIP_RANGE[1])
        
        img = ecg_to_contour_image(signal_500, target_width=TARGET_WIDTH, clip_range=CLIP_RANGE)
        
        if fs != 100:
            signal_100, _ = resample_ecg(signal, fs_in=fs, fs_out=100)
        else:
            signal_100 = signal
        signal_100 = pad_or_trim(signal_100, TARGET_SAMPLES_100)
        
        # Save
        rec_id = rec_path.name
        np.save(IMG_BASE / ds / f"{rec_id}_img.npy", img.astype(np.float32))
        np.save(FM_BASE / ds / f"{rec_id}_fm.npy", signal_100.astype(np.float32))
        
        return rec_id, None  # Success

    except Exception as e:
        return rec_path.name, str(e)  # Failure

def build_images_for_dataset(ds, subset):
    wfdb_dir = WFDB_BASE / ds
    out_img_dir = IMG_BASE / ds
    out_fm_dir = FM_BASE / ds
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_fm_dir.mkdir(parents=True, exist_ok=True)
    
    record_paths = []
    for root, _, files in os.walk(wfdb_dir):
        for f in files:
            if f.endswith('.hea'):
                record_paths.append(Path(root) / f.replace('.hea', ''))
    
    if subset < 1.0:
        record_paths = random.sample(record_paths, int(len(record_paths) * subset))
    
    print(f"\n📂 {ds.upper()}: {len(record_paths)} records")
    
    args_list = [(ds, rp) for rp in record_paths]
    
    processed = 0
    failed = 0
    errors = []
    
    with Pool(N_JOBS) as p:
        results = list(tqdm(p.imap(process_record, args_list), total=len(record_paths), desc="Generating images/FM"))
    
    for rec_id, err in results:
        if err is None:
            processed += 1
        else:
            failed += 1
            errors.append(f"{rec_id}: {err}")
    
    print(f" ✅ Saved: {processed} | ❌ Failed: {failed}")
    if errors:
        print("\n❌ Errors:")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more")
    
    return processed, failed

def main(subset):
    print("Starting processing...")
    
    print("\n" + "=" * 70)
    print("🚀 Building 2D Contour Images from Official WFDB Data (Kim et al. 2025)")
    print("=" * 70 + "\n")
    
    IMG_BASE.mkdir(parents=True, exist_ok=True)
    FM_BASE.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_failed = 0
    
    for ds in DATASETS:
        proc, fail = build_images_for_dataset(ds, subset)
        total_processed += proc
        total_failed += fail
    
    print("\n" + "=" * 70)
    print("📈 FINAL SUMMARY")
    print("=" * 70)
    print(f" Total Images/FM Generated: {total_processed:,}")
    print(f" Total Failed : {total_failed}")
    print(f" Output Location Images : {IMG_BASE}")
    print(f" Output Location FM : {FM_BASE}")
    print("=" * 70)
    print("✅ 2D Image and FM Generation Complete! Ready for ViT/FM training.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2D images and FM signals from WFDB")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction to process (0-1)")
    args = parser.parse_args()
    main(args.subset)