"""Master script to build all processed data (all bugs fixed).

This script:
  1. Loads raw WFDB files for PTB-XL, SaMi-Trop, and CODE-15%.
  2. Applies dataset-specific baseline removal.
  3. Depads SaMi-Trop and CODE-15% where necessary.
  4. Pads/trims to 10 seconds at original sampling rate.
  5. Generates 2D contour images: (3, 24, 2048) uint8.
  6. Generates 1D signals for FM: (12, 1000) float32 at 100 Hz.
  7. Saves per-dataset and combined metadata CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_signal, pad_or_trim
from src.preprocessing.normalization import normalize_per_lead
from src.preprocessing.image_embedding import build_2d_image


# Dataset-specific baseline configuration (corrected)
BASELINE_CONFIG = {
    "ptbxl": {
        "method": "bandpass",
        "low_cut_hz": 0.5,
        "high_cut_hz": 40.0,  # Fixed from 45.0
        "order": 4,
    },
    "sami_trop": {
        "method": "highpass",  # Fixed from moving_average
        "cutoff_hz": 0.5,
        "order": 3,
    },
    "code15": {
        "method": "bandpass",  # Fixed from None
        "low_cut_hz": 0.5,
        "high_cut_hz": 40.0,
        "order": 4,
    },
}


def process_single_record(
    record_path: Path,
    dataset_name: str,
    output_dir_2d: Path,
    output_dir_1d: Path,
    label: float,
    age=None,
    sex=None,
    record_id: str | None = None,
):
    """Process a single ECG record through the complete pipeline.

    Returns:
        dict with metadata or None if failed
    """

    try:
        # 1. Load WFDB record
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal.T  # (num_leads, num_samples)
        original_fs = record.fs

        # Ensure 12 leads
        if signal.shape[0] != 12:
            print(f"Warning: {record_path} has {signal.shape[0]} leads, skipping")
            return None

        # 2. Baseline removal (dataset-specific)
        baseline_config = BASELINE_CONFIG[dataset_name]
        signal = remove_baseline(signal, fs=original_fs, **baseline_config)

        # 3. Depadding (for SaMi-Trop and CODE-15%)
        if dataset_name in ["sami_trop", "code15"]:
            non_zero_mask = np.any(signal != 0, axis=0)
            if not np.all(non_zero_mask):
                last_nonzero = np.where(non_zero_mask)[0][-1]
                signal = signal[:, : last_nonzero + 1]

        # 4. Pad/trim to 10 seconds at original sampling rate
        target_samples_orig = int(10 * original_fs)
        signal = pad_or_trim(signal, target_samples_orig)

        # ===== 2D PATHWAY =====
        # 5a. Resample to 500 Hz
        signal_500hz = resample_signal(signal, original_fs, 500)

        # 6a. Normalize per-lead (z-score with clipping)
        signal_500hz_norm = normalize_per_lead(
            signal_500hz, method="zscore", clip_std=3.0
        )

        # 7a. Build 2D image
        img_2d = build_2d_image(
            signal_500hz_norm, target_height=24, target_width=2048
        )

        assert img_2d.dtype == np.uint8, f"Image must be uint8, got {img_2d.dtype}"
        assert img_2d.shape == (
            3,
            24,
            2048,
        ), f"Image shape must be (3,24,2048), got {img_2d.shape}"

        # 8a. Save 2D image
        img_filename = f"{record_id}.npy" if record_id else f"{record_path.stem}.npy"
        img_path = output_dir_2d / img_filename
        np.save(img_path, img_2d)

        # ===== 1D PATHWAY =====
        # 5b. Resample to 100 Hz
        signal_100hz = resample_signal(signal, original_fs, 100)

        # 6b. Normalize per-lead (z-score, no clipping)
        signal_100hz_norm = normalize_per_lead(
            signal_100hz, method="zscore", clip_std=None
        )

        # 7b. Ensure exactly 1000 samples (10 s × 100 Hz)
        signal_100hz_norm = pad_or_trim(signal_100hz_norm, 1000)

        assert signal_100hz_norm.dtype == np.float32
        assert signal_100hz_norm.shape == (12, 1000)

        # 8b. Save 1D signal
        sig_filename = img_filename
        sig_path = output_dir_1d / sig_filename
        np.save(sig_path, signal_100hz_norm)

        return {
            "id": record_id or record_path.stem,
            "dataset": dataset_name,
            "label": float(label),
            "img_path": str(img_path),
            "fm_path": str(sig_path),
            "age": age,
            "sex": sex,
        }

    except Exception as e:  # noqa: BLE001
        print(f"Error processing {record_path}: {e}")
        return None


def process_ptbxl(ptbxl_dir: Path, output_dir_2d: Path, output_dir_1d: Path, subset: float = 1.0):
    """Process PTB-XL dataset (all assumed Chagas-negative)."""

    print("\n" + "=" * 60)
    print("PROCESSING PTB-XL")
    print("=" * 60)

    metadata_path = ptbxl_dir / "ptbxl_database.csv"
    df = pd.read_csv(metadata_path)

    df["chagas_label"] = 0

    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    print(f"Total records: {len(df)}")
    print(f"Chagas positive: {df['chagas_label'].sum()}")

    output_dir_2d.mkdir(parents=True, exist_ok=True)
    output_dir_1d.mkdir(parents=True, exist_ok=True)

    metadata_list: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL"):
        record_path = ptbxl_dir / row["filename_hr"].replace(".hea", "")

        metadata = process_single_record(
            record_path=record_path,
            dataset_name="ptbxl",
            output_dir_2d=output_dir_2d,
            output_dir_1d=output_dir_1d,
            label=row["chagas_label"],
            age=row.get("age", None),
            sex=row.get("sex", None),
            record_id=f"ptbxl_{row['ecg_id']}",
        )

        if metadata is not None:
            metadata_list.append(metadata)

    print(f"\nSuccessfully processed: {len(metadata_list)}/{len(df)}")
    return metadata_list


def process_samitrop(
    samitrop_dir: Path,
    output_dir_2d: Path,
    output_dir_1d: Path,
    subset: float = 1.0,
):
    """Process SaMi-Trop dataset (all Chagas-positive in training)."""

    print("\n" + "=" * 60)
    print("PROCESSING SAMI-TROP")
    print("=" * 60)

    metadata_path = samitrop_dir / "exams.csv"
    df = pd.read_csv(metadata_path)

    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    print(f"Total records: {len(df)}")
    print(f"Chagas positive: {df['chagas'].sum()}")

    output_dir_2d.mkdir(parents=True, exist_ok=True)
    output_dir_1d.mkdir(parents=True, exist_ok=True)

    metadata_list: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SaMi-Trop"):
        record_path = samitrop_dir / "wfdb" / str(row["exam_id"])

        metadata = process_single_record(
            record_path=record_path,
            dataset_name="sami_trop",
            output_dir_2d=output_dir_2d,
            output_dir_1d=output_dir_1d,
            label=row["chagas"],
            age=row.get("age", None),
            sex=row.get("sex", None),
            record_id=f"samitrop_{row['exam_id']}",
        )

        if metadata is not None:
            metadata_list.append(metadata)

    print(f"\nSuccessfully processed: {len(metadata_list)}/{len(df)}")
    return metadata_list


def process_code15(
    code15_dir: Path,
    output_dir_2d: Path,
    output_dir_1d: Path,
    subset: float = 1.0,
):
    """Process CODE-15% dataset (full 15% subset with Chagas labels)."""

    print("\n" + "=" * 60)
    print("PROCESSING CODE-15%")
    print("=" * 60)

    metadata_path = code15_dir / "code15_chagas_labels.csv"
    df = pd.read_csv(metadata_path)

    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    print(f"Total records: {len(df)}")
    print(f"Chagas positive: {df['chagas'].sum()}")

    output_dir_2d.mkdir(parents=True, exist_ok=True)
    output_dir_1d.mkdir(parents=True, exist_ok=True)

    metadata_list: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CODE-15%"):
        record_path = code15_dir / "wfdb" / str(row["exam_id"])

        metadata = process_single_record(
            record_path=record_path,
            dataset_name="code15",
            output_dir_2d=output_dir_2d,
            output_dir_1d=output_dir_1d,
            label=row["chagas"],
            age=row.get("age", None),
            sex=row.get("sex", None),
            record_id=f"code15_{row['exam_id']}",
        )

        if metadata is not None:
            metadata_list.append(metadata)

    print(f"\nSuccessfully processed: {len(metadata_list)}/{len(df)}")
    return metadata_list


def main() -> None:
    """Main preprocessing pipeline entry point."""

    project_root = Path(__file__).parent.parent
    data_root = project_root / "data"

    official_wfdb = data_root / "official_wfdb"
    processed_root = data_root / "processed"

    output_2d = processed_root / "2d_images"
    output_1d = processed_root / "1d_signals_100hz"
    metadata_dir = processed_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_metadata: list[dict] = []

    # PTB-XL
    if (official_wfdb / "ptbxl").exists():
        ptbxl_metadata = process_ptbxl(
            ptbxl_dir=official_wfdb / "ptbxl",
            output_dir_2d=output_2d / "ptbxl",
            output_dir_1d=output_1d / "ptbxl",
            subset=1.0,
        )
        all_metadata.extend(ptbxl_metadata)
        pd.DataFrame(ptbxl_metadata).to_csv(
            metadata_dir / "ptbxl_metadata.csv", index=False
        )

    # SaMi-Trop
    if (official_wfdb / "sami_trop").exists():
        samitrop_metadata = process_samitrop(
            samitrop_dir=official_wfdb / "sami_trop",
            output_dir_2d=output_2d / "sami_trop",
            output_dir_1d=output_1d / "sami_trop",
            subset=1.0,
        )
        all_metadata.extend(samitrop_metadata)
        pd.DataFrame(samitrop_metadata).to_csv(
            metadata_dir / "sami_trop_metadata.csv", index=False
        )

    # CODE-15% (full, not re-balanced)
    if (official_wfdb / "code15").exists():
        code15_metadata = process_code15(
            code15_dir=official_wfdb / "code15",
            output_dir_2d=output_2d / "code15",
            output_dir_1d=output_1d / "code15",
            subset=1.0,
        )
        all_metadata.extend(code15_metadata)
        pd.DataFrame(code15_metadata).to_csv(
            metadata_dir / "code15_metadata.csv", index=False
        )

    all_df = pd.DataFrame(all_metadata)
    all_df.to_csv(metadata_dir / "all_data.csv", index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nTotal samples: {len(all_df)}")
    print(
        f"Chagas positive: {all_df['label'].sum()} "
        f"({all_df['label'].mean() * 100:.2f}%)"
    )
    print("\nBy dataset:")
    print(all_df["dataset"].value_counts())

    print(f"\n2D images saved to: {output_2d}")
    print(f"1D signals saved to: {output_1d}")
    print(f"Metadata saved to: {metadata_dir}")

    img_size_kb = 3 * 24 * 2048 / 1024  # 3*24*2048 uint8
    sig_size_kb = 12 * 1000 * 4 / 1024  # 12*1000 float32
    total_storage_gb = len(all_df) * (img_size_kb + sig_size_kb) / 1024 / 1024
    print(f"\nEstimated storage: {total_storage_gb:.2f} GB")


if __name__ == "__main__":
    main()
