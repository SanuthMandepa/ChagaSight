# veriffy_data.py
# Validate outputs: files exist, shapes/dtypes correct, soft-label ranges.

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _check_npy(path: Path, expected_shape, expected_dtype):
    arr = np.load(path, allow_pickle=False)
    if arr.shape != expected_shape:
        raise ValueError(f"{path}: shape {arr.shape} != expected {expected_shape}")
    if arr.dtype != expected_dtype:
        raise ValueError(f"{path}: dtype {arr.dtype} != expected {expected_dtype}")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", type=str, default="data/processed/metadata/all_data.csv")
    p.add_argument("--num_samples", type=int, default=50)
    args = p.parse_args()

    df = pd.read_csv(args.metadata_csv)
    if len(df) == 0:
        raise ValueError("Metadata CSV is empty (0 samples).")

    required = ["id", "dataset", "label_hard", "label_soft", "img_path", "fm_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Label sanity
    if not df["label_hard"].isin([0, 1]).all():
        bad = df.loc[~df["label_hard"].isin([0, 1]), "label_hard"].head(10).tolist()
        raise ValueError(f"Found non-binary hard labels, examples: {bad}")

    if not ((df["label_soft"] >= 0.0) & (df["label_soft"] <= 1.0)).all():
        bad = df.loc[~((df["label_soft"] >= 0.0) & (df["label_soft"] <= 1.0)), "label_soft"].head(10).tolist()
        raise ValueError(f"Found soft labels outside [0,1], examples: {bad}")

    # Spot-check samples
    n = min(args.num_samples, len(df))
    sample_df = df.sample(n=n, random_state=42)

    for _, row in sample_df.iterrows():
        img_path = Path(row["img_path"])
        fm_path = Path(row["fm_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not fm_path.exists():
            raise FileNotFoundError(f"Missing FM signal: {fm_path}")

        _check_npy(img_path, expected_shape=(3, 24, 2048), expected_dtype=np.uint8)
        _check_npy(fm_path, expected_shape=(12, 1000), expected_dtype=np.float32)

    print("Verification passed.")
    print("Dataset counts:\n", df["dataset"].value_counts())
    print("Hard positives:", int(df["label_hard"].sum()))
    print("Mean soft label:", float(df["label_soft"].mean()))


if __name__ == "__main__":
    main()
