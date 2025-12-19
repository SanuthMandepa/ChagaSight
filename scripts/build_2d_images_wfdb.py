"""
Build structured 2D ECG images (3 × 24 × 2048) directly from official WFDB files.

Pipeline:
1. Load WFDB signal
2. Baseline removal
3. Resample to 500 Hz
4. Z-score normalization per lead
5. Clip to [-3, 3]
6. RA/LA/LL contour embedding (Kim et al.)

Handles PTB-XL subfolders.
Parallelized for speed.
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

# Fix import path for module run (-m)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg
from src.preprocessing.normalization import normalize_dataset
from src.preprocessing.image_embedding import ecg_to_contour_image

# Configuration
DATASETS = ["ptbxl", "sami_trop"]  # Run per-dataset to avoid crashes; change as needed
WFDB_BASE = Path("data/official_wfdb")
IMG_BASE = Path("data/processed/2d_images")
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)
N_JOBS = 2  # Lower default for full runs (RAM-safe; adjust to 6 for smaller datasets)

def process_record(args):
    ds, rec_path = args
    try:
        signal, fields = wfdb.rdsamp(str(rec_path))
        fs = fields['fs']
        
        signal = remove_baseline(signal, fs=fs)
        signal_500, _ = resample_ecg(signal, fs_in=fs, fs_out=500)
        signal_norm = normalize_dataset(signal_500)
        img = ecg_to_contour_image(signal_norm, target_width=TARGET_WIDTH, clip_range=CLIP_RANGE)
        
        return np.clip(img, *CLIP_RANGE).astype(np.float32), rec_path.name
    except Exception as e:
        return None, str(e)

def build_images_for_dataset(dataset: str, subset=1.0):
    wfdb_dir = WFDB_BASE / dataset
    out_dir = IMG_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    record_paths = []
    for root, _, files in os.walk(wfdb_dir):
        for f in files:
            if f.endswith('.hea'):
                rec_path = Path(root) / f.replace('.hea', '')
                record_paths.append(rec_path)
    
    if subset < 1.0:
        num_to_process = int(len(record_paths) * subset)
        record_paths = random.sample(record_paths, num_to_process)
        print(f"Processing subset: {subset*100:.1f}% ({num_to_process} records)")
    
    print(f"\n📂 {dataset.upper()}: {len(record_paths)} records")
    
    args_list = [(dataset, rp) for rp in record_paths]
    
    processed = 0
    failed = 0
    
    with Pool(N_JOBS) as p:
        results = list(tqdm(p.imap(process_record, args_list), total=len(args_list), desc="Generating images"))
    
    for img, msg in results:
        if img is not None:
            rec_id = msg
            np.save(out_dir / f"{rec_id}_img.npy", img)
            processed += 1
        else:
            failed += 1
    
    print(f" ✅ Saved: {processed} | ❌ Failed: {failed}")
    return processed, failed

def main(subset):
    print("Starting processing...")  # Immediate feedback
    
    print("\n" + "=" * 70)
    print("🚀 Building 2D Contour Images from Official WFDB Data (Kim et al. 2025)")
    print("=" * 70 + "\n")
    
    IMG_BASE.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_failed = 0
    
    for ds in DATASETS:
        proc, fail = build_images_for_dataset(ds, subset)
        total_processed += proc
        total_failed += fail
    
    print("\n" + "=" * 70)
    print("📈 FINAL SUMMARY")
    print("=" * 70)
    print(f" Total Images Generated: {total_processed:,}")
    print(f" Total Failed : {total_failed}")
    print(f" Output Location : {IMG_BASE}")
    print("=" * 70)
    print("✅ Ready for Hybrid Vision Transformer Training!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2D images from WFDB")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of dataset to process (0-1)")
    args = parser.parse_args()
    main(args.subset)