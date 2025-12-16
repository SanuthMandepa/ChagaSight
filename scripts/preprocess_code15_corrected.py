# scripts/preprocess_code15_corrected.py
"""Correct CODE-15% preprocessing - signals are already filtered & normalized."""

import os
import time
import argparse
import warnings
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.preprocessing.resample import resample_ecg, pad_or_trim

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
RAW_DIR = Path("data/raw/code15")
OUTPUT_DIR = Path("data/processed/1d_signals_100hz/code15")  # For foundation model
OUTPUT_DIR_400 = Path("data/processed/1d_signals_400hz/code15")  # For image model

# For foundation model (100 Hz)
TARGET_FS_FM = 100.0  # Foundation model expects 100 Hz
TARGET_DURATION_FM = 10.0
TARGET_SAMPLES_FM = int(TARGET_DURATION_FM * TARGET_FS_FM)  # 1000 samples

# For image model (400 Hz - but papers use 500 Hz, Kim et al. resamples to 500 Hz)
TARGET_FS_IMAGE = 400.0  # Keep at 400 Hz (Kim et al. uses 500 Hz)
TARGET_DURATION_IMAGE = 10.0
TARGET_SAMPLES_IMAGE = int(TARGET_DURATION_IMAGE * TARGET_FS_IMAGE)  # 4000 samples

def remove_zero_padding(signal):
    """
    Remove zero-padding from CODE-15% signals.
    
    CODE-15% pads 7s signals (2800 samples) with zeros to 4096 samples.
    We need to find and remove this padding.
    """
    T = signal.shape[0]
    
    # Check if this is a 7s signal (2800 samples) padded to 4096
    # Find non-zero region
    signal_energy = np.sum(np.abs(signal), axis=1)
    
    # Find first and last non-zero samples
    non_zero_idx = np.where(signal_energy > 1e-6)[0]
    
    if len(non_zero_idx) == 0:
        return signal  # All zeros, return as-is
    
    start_idx = non_zero_idx[0]
    end_idx = non_zero_idx[-1] + 1  # Exclusive
    
    # If it looks like a 7s signal (around 2800 samples)
    signal_length = end_idx - start_idx
    if 2700 <= signal_length <= 2900:
        # Extract the actual 7s signal
        unpadded = signal[start_idx:end_idx, :]
        
        # Now we need to handle this 7s signal
        # Options:
        # 1. Pad to 10s (4000 samples) at 400 Hz
        # 2. Resample 7s → 10s (not ideal)
        # According to Kim et al. paper, they pad shorter signals
        return unpadded
    else:
        # Probably a 10s signal (4000 samples) padded to 4096
        # Remove symmetric padding
        pad_total = 4096 - T
        if pad_total > 0:
            pad_start = pad_total // 2
            pad_end = pad_total - pad_start
            return signal[pad_start:-pad_end, :]
        return signal

def main(subset_fraction: float):
    print("===================================================")
    print("⏱️  Starting CORRECT CODE-15% Preprocessing")
    print("Based on dataset documentation: Already filtered & normalized")
    print("===================================================\n")

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_400.mkdir(parents=True, exist_ok=True)

    # Load CSV metadata
    csv_path = RAW_DIR / "exams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CODE-15% metadata not found at {csv_path}")
    
    df = pd.read_csv(csv_path)

    print(f"📄 Loaded metadata: {csv_path}")
    print(f"🔢 Total ECG exams found: {len(df)}")
    print(f"ℹ️  Signals are already: Band-pass filtered (0.5-40 Hz) + Z-score normalized")

    # SUBSET MODE
    original_count = len(df)
    if subset_fraction < 1.0:
        df = df.sample(frac=subset_fraction, random_state=42)
        print(f"⚠️ SUBSET MODE → Using {len(df)} exams ({subset_fraction*100:.1f}% of {original_count})")

    total_exams = len(df)
    print(f"🔢 Total exams to process: {total_exams}")

    part_groups = df.groupby("trace_file")
    total_processed = 0
    total_failed = 0
    
    processed_records = []

    for trace_file, group in part_groups:
        shard_start = time.time()
        h5_path = RAW_DIR / trace_file

        if not h5_path.exists():
            warnings.warn(f"HDF5 file not found: {h5_path}")
            continue

        print(f"\n📂 Opening HDF5 file: {trace_file}  | Exams: {len(group)}")

        with h5py.File(h5_path, "r") as h5:
            exam_ids = np.array(h5["exam_id"])
            tracings = h5["tracings"]
            
            index_map = {int(exam_ids[i]): i for i in range(len(exam_ids))}

            for _, row in tqdm(group.iterrows(), total=len(group), desc=f"Processing {trace_file}"):
                exam_id = int(row["exam_id"])

                if exam_id not in index_map:
                    print(f"⚠️ Missing exam_id {exam_id} in {trace_file}")
                    total_failed += 1
                    continue

                idx = index_map[exam_id]
                
                if idx >= tracings.shape[0]:
                    print(f"⚠️ Index {idx} out of bounds for {trace_file}")
                    total_failed += 1
                    continue

                try:
                    # Load raw signal (4096, 12)
                    raw = np.asarray(tracings[idx, :, :])
                    
                    # 1. Remove zero-padding (4096 → 4000 or 2800)
                    unpadded = remove_zero_padding(raw)
                    
                    # 2. NO BASELINE REMOVAL - signals are already filtered
                    
                    # 3. Process for foundation model (100 Hz)
                    # Resample to 100 Hz if needed
                    if unpadded.shape[0] / 400.0 < 10.0:
                        # Short signal (7s), pad to 10s at original fs
                        target_samples_original_fs = int(10.0 * 400.0)
                        padded = pad_or_trim(unpadded, target_samples_original_fs)
                    else:
                        padded = pad_or_trim(unpadded, 4000)  # 10s at 400 Hz
                    
                    # Resample to 100 Hz
                    resampled_100hz, fs_100 = resample_ecg(
                        padded, 
                        fs_in=400.0, 
                        fs_out=TARGET_FS_FM
                    )
                    
                    # Trim to exactly 1000 samples (10s at 100 Hz)
                    final_100hz = pad_or_trim(resampled_100hz, TARGET_SAMPLES_FM)
                    
                    # 4. Process for image model (400 Hz)
                    # Already at 400 Hz, just ensure 4000 samples
                    final_400hz = pad_or_trim(unpadded, TARGET_SAMPLES_IMAGE)
                    
                    # 5. NO Z-SCORE - signals are already normalized
                    # But we should verify and optionally re-normalize
                    # Check if signal is already normalized
                    mean_per_lead = final_100hz.mean(axis=0)
                    std_per_lead = final_100hz.std(axis=0)
                    
                    if np.any(np.abs(mean_per_lead) > 0.5) or np.any(np.abs(std_per_lead - 1.0) > 0.5):
                        # Not properly normalized, apply z-score
                        from src.preprocessing.normalization import zscore_per_lead
                        final_100hz = zscore_per_lead(final_100hz)
                        final_400hz = zscore_per_lead(final_400hz)
                    
                    # 6. Save both versions
                    # Foundation model version (100 Hz)
                    out_path_fm = OUTPUT_DIR / f"{exam_id}.npy"
                    np.save(out_path_fm, final_100hz)
                    
                    # Image model version (400 Hz)
                    out_path_image = OUTPUT_DIR_400 / f"{exam_id}.npy"
                    np.save(out_path_image, final_400hz)
                    
                    # Track metadata
                    processed_records.append({
                        "exam_id": exam_id,
                        "trace_file": trace_file,
                        "original_shape": str(raw.shape),
                        "unpadded_shape": str(unpadded.shape),
                        "fm_path": str(out_path_fm),
                        "fm_shape": str(final_100hz.shape),
                        "fm_fs": TARGET_FS_FM,
                        "image_path": str(out_path_image),
                        "image_shape": str(final_400hz.shape),
                        "image_fs": TARGET_FS_IMAGE,
                        "duration_sec": TARGET_DURATION_FM,
                        "note": "Already filtered & normalized per dataset documentation"
                    })
                    
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        print(f"   Processed {total_processed}/{total_exams} exams")

                except Exception as e:
                    print(f"❌ ERROR processing exam_id {exam_id}: {e}")
                    total_failed += 1
                    continue

        shard_elapsed = time.time() - shard_start
        print(f"⏳ Shard completed in {shard_elapsed:.1f} sec")

    # Save metadata
    if processed_records:
        metadata_df = pd.DataFrame(processed_records)
        metadata_path = Path("data/processed/metadata/code15_processed_corrected.csv")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\n💾 Saved metadata to {metadata_path}")

    # FINISHED
    elapsed = time.time() - t0
    
    print("\n" + "="*50)
    print("🎉 CORRECT CODE-15% preprocessing complete!")
    print("="*50)
    print(f"⏱️ Total time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")
    print(f"📦 Total processed signals: {total_processed}")
    print(f"❌ Total failed signals: {total_failed}")
    print(f"📈 Success rate: {total_processed/max(1, total_processed+total_failed)*100:.1f}%")
    print(f"\n💾 Saved to:")
    print(f"   Foundation model (100 Hz): {OUTPUT_DIR}")
    print(f"   Image model (400 Hz): {OUTPUT_DIR_400}")
    print(f"\nℹ️  Key changes:")
    print(f"   1. NO baseline removal (signals already filtered)")
    print(f"   2. Removed zero-padding (4096 → 4000/2800 samples)")
    print(f"   3. Dual output: 100 Hz for FM, 400 Hz for images")
    print(f"   4. Optional z-score only if needed")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=0.01, help="Fraction of CODE-15 dataset to preprocess")
    args = parser.parse_args()
    main(args.subset)