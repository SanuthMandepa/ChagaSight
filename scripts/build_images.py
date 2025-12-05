"""
Build structured 2D ECG images (3 × 24 × 2048) from preprocessed 1D signals.

Uses RA/LA/LL contour mapping from:
    src/preprocessing/image_embedding.py

Input:
    data/processed/1d_signals/<dataset>/<id>.npy

Output:
    data/processed/2d_images/<dataset>/<id>_img.npy
"""

import os
import glob
import numpy as np
from tqdm import tqdm

from src.preprocessing.image_embedding import ecg_to_contour_image


# ---------------------------------------------------------
# DATASET LIST
# ---------------------------------------------------------
DATASETS = ["ptbxl", "sami_trop", "code15"]

BASE_IN = "data/processed/1d_signals"
BASE_OUT = "data/processed/2d_images"


def build_images_for_dataset(dataset: str):
    in_dir = os.path.join(BASE_IN, dataset)
    out_dir = os.path.join(BASE_OUT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    npy_files = glob.glob(os.path.join(in_dir, "*.npy"))
    print(f"\n📂 Dataset: {dataset} | {len(npy_files)} signals")

    for path in tqdm(npy_files):
        try:
            signal = np.load(path)              # (4000, 12)
            img = ecg_to_contour_image(signal)  # (3, 24, 2048)

            out_path = path.replace("1d_signals", "2d_images").replace(".npy", "_img.npy")
            np.save(out_path, img)

        except Exception as e:
            print(f"❌ Failed on {path}: {e}")


def main():
    print("\n===============================================")
    print("🚀 Building RA/LA/LL structured ECG images...")
    print("===============================================\n")

    for ds in DATASETS:
        build_images_for_dataset(ds)

    print("\n🎉 DONE — All datasets converted to 2D images!")
    print("Saved inside: data/processed/2d_images/\n")


if __name__ == "__main__":
    main()
