"""
Build structured 2D ECG images (3 × 24 × 2048) from preprocessed 1D signals.

Uses RA/LA/LL contour mapping from:
    src/preprocessing/image_embedding.py

Input (ONLY 500 Hz signals – z-scored, for images):
    data/processed/1d_signals_500hz/<dataset>/<id>.npy

Output:
    data/processed/2d_images/<dataset>/<id>_img.npy
"""

import time
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.preprocessing.image_embedding import ecg_to_contour_image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATASETS = ["ptbxl", "sami_trop", "code15"]

# Use ONLY the 500 Hz z-scored signals (Kim et al. 2025)
BASE_IN = Path("data/processed/1d_signals_500hz")
BASE_OUT = Path("data/processed/2d_images")

# Image generation parameters (Kim et al.)
TARGET_WIDTH = 2048
CLIP_RANGE = (-3.0, 3.0)  # Standard for z-scored signals


def build_images_for_dataset(dataset: str):
    """
    Convert all 500 Hz 1D signals in a dataset to 2D contour images.
    """
    in_dir = BASE_IN / dataset
    out_dir = BASE_OUT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(in_dir.glob("*.npy"))
    if not npy_files:
        print(f"⚠️  No signals found in {in_dir}")
        return 0, 0

    print(f"\n📂 Dataset: {dataset.upper()}")
    print(f"   Input : {in_dir} ({len(npy_files)} files)")
    print(f"   Output: {out_dir}")

    processed_count = 0
    failed_count = 0
    start_time = time.time()

    for npy_path in tqdm(npy_files, desc=f"Converting {dataset}"):
        file_id = npy_path.stem
        img_path = out_dir / f"{file_id}_img.npy"

        # Skip if already exists
        if img_path.exists():
            processed_count += 1
            continue

        try:
            signal = np.load(npy_path)

            # Shape validation
            if signal.shape[1] != 12:
                raise ValueError(f"Expected 12 leads, got {signal.shape}")

            # Generate image
            img = ecg_to_contour_image(
                signal=signal,
                target_width=TARGET_WIDTH,
                clip_range=CLIP_RANGE,
                check_normalization=True
            )

            if img.shape != (3, 24, TARGET_WIDTH):
                warnings.warn(f"Unexpected image shape {img.shape} for {file_id}")

            np.save(img_path, img)
            processed_count += 1

        except Exception as e:
            tqdm.write(f"❌ Failed {file_id}: {e}")
            failed_count += 1

    elapsed = time.time() - start_time
    print(f"   ✅ Processed: {processed_count} | ❌ Failed: {failed_count}")
    print(f"   ⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return processed_count, failed_count


def main():
    print("\n" + "=" * 60)
    print("🚀 Building RA/LA/LL Structured 2D ECG Images (Kim et al. 2025)")
    print("=" * 60 + "\n")

    total_processed = 0
    total_failed = 0

    for dataset in DATASETS:
        processed, failed = build_images_for_dataset(dataset)
        total_processed += processed
        total_failed += failed

    print("\n" + "=" * 60)
    print("📈 FINAL SUMMARY")
    print("=" * 60)
    print(f"   Total Images Generated: {total_processed}")
    print(f"   Total Failed          : {total_failed}")
    print(f"   Output Location       : {BASE_OUT}")
    print("=" * 60)
    print("✅ 2D Image Generation Complete! Ready for Vision Transformer training.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()