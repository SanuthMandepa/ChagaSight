"""Verify preprocessed data quality."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def verify_data() -> None:
    """Verify all preprocessed data (shapes, dtypes, ranges)."""

    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    metadata_path = Path("data/processed/metadata/all_data_5fold.csv")
    if not metadata_path.exists():
        print("Metadata not found. Run create_splits.py first.")
        return

    df = pd.read_csv(metadata_path)
    print(f"\nTotal samples: {len(df)}")

    errors: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
        # 2D image
        img_path = Path(row["img_path"])
        if not img_path.exists():
            errors.append(f"Missing image: {img_path}")
            continue

        try:
            img = np.load(img_path, mmap_mode="r")

            if img.shape != (3, 24, 2048):
                errors.append(f"Wrong image shape: {img_path} has {img.shape}")

            if img.dtype != np.uint8:
                errors.append(
                    f"Wrong image dtype: {img_path} is {img.dtype}, should be uint8"
                )

            if img.min() < 0 or img.max() > 255:
                errors.append(
                    f"Image out of range: {img_path} has "
                    f"[{img.min()}, {img.max()}]"
                )

        except Exception as e:  # noqa: BLE001
            errors.append(f"Error loading image {img_path}: {e}")

        # 1D signal
        sig_path = Path(row["fm_path"])
        if not sig_path.exists():
            errors.append(f"Missing signal: {sig_path}")
            continue

        try:
            sig = np.load(sig_path, mmap_mode="r")

            if sig.shape != (12, 1000):
                errors.append(f"Wrong signal shape: {sig_path} has {sig.shape}")

            if sig.dtype != np.float32:
                errors.append(
                    f"Wrong signal dtype: {sig_path} is {sig.dtype}, "
                    "should be float32",
                )

        except Exception as e:  # noqa: BLE001
            errors.append(f"Error loading signal {sig_path}: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    if len(errors) == 0:
        print("All checks passed.")
        print(f"{len(df)} samples verified successfully.")

        sample_img = np.load(df.iloc[0]["img_path"], mmap_mode="r")
        sample_sig = np.load(df.iloc[0]["fm_path"], mmap_mode="r")

        img_size_kb = sample_img.nbytes / 1024
        sig_size_kb = sample_sig.nbytes / 1024

        print("\nData statistics:")
        print(f"  2D image size: {img_size_kb:.1f} KB per sample")
        print(f"  1D signal size: {sig_size_kb:.1f} KB per sample")
        print(
            f"  Total storage: "
            f"{(len(df) * (img_size_kb + sig_size_kb)) / 1024:.1f} MB"
        )

    else:
        print(f"Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    verify_data()
