# scripts/build_all_data.py - v2 (pad-before-filter fix)
# Features:
# - Dataset selection: --datasets ptbxl samitrop code15
# - Smart CSV merging: skips reprocessing existing datasets
# - Automatic metadata combination
# - FIX: pad to target length BEFORE baseline filter so very short
#   records (e.g. 26-sample code15 files) are no longer silently dropped

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.preprocessing.baseline_removal import BaselineConfig, remove_baseline
from src.preprocessing.resample import resample_signal, pad_or_trim
from src.preprocessing.normalization import normalize_per_lead
from src.preprocessing.image_embedding import build_2d_image, STANDARD_12
from src.preprocessing.soft_labels import hard_to_soft_label


# -------------------- helpers for sex -------------------- #

def _normalize_sex_value(raw: Any) -> Optional[int]:
    """
    Normalize various sex encodings to 1/0:
      1 = male, 0 = female, None = unknown/missing.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None

    if isinstance(raw, bool):
        return 1 if raw else 0

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) == 1:
            return 1
        if int(raw) == 0:
            return 0

    s = str(raw).strip().lower()
    if s in {"m", "male", "1", "true", "t", "yes", "y"}:
        return 1
    if s in {"f", "female", "0", "false", "no", "n"}:
        return 0

    return None


def _try_import_helper_code(helper_dir: Optional[str]):
    if helper_dir:
        sys.path.insert(0, helper_dir)
    try:
        import helper_code  # type: ignore
        return helper_code
    except Exception:
        return None


def _reorder_to_standard(
    psignal: np.ndarray,  # (T, C)
    sig_names: List[str],
    helper_code_module,
) -> np.ndarray:
    if helper_code_module is None:
        return psignal.T.astype(np.float32, copy=False)
    try:
        reordered = helper_code_module.reordersignal(psignal, sig_names, STANDARD_12)
        return reordered.T.astype(np.float32, copy=False)
    except Exception:
        return psignal.T.astype(np.float32, copy=False)


def _depad_trailing_zeros(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError("signal must be (L,T)")
    non_zero_mask = np.any(signal != 0.0, axis=0)
    if not np.any(non_zero_mask):
        return signal
    last = int(np.where(non_zero_mask)[0][-1])
    return signal[:, : last + 1]


_skip_log: List[str] = []   # collects skip reasons; printed at dataset end


def process_single_record(
    record_path: Path,
    dataset: str,
    out_2d_dir: Path,
    out_1d_dir: Path,
    label_hard: int,
    label_soft: float,
    age: Optional[float],
    sex: Optional[int],
    helper_code_module=None,
    baseline_cfg: Optional[BaselineConfig] = None,
    make_random_crop: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    try:
        record = wfdb.rdrecord(str(record_path))
        psignal = record.p_signal
        sig_names = list(getattr(record, "sig_name", []))

        signal = _reorder_to_standard(psignal, sig_names, helper_code_module)
        fs = float(record.fs)

        if signal.shape[0] != 12:
            _skip_log.append(
                f"SKIP {record_path.name}: expected 12 leads, got {signal.shape[0]}"
            )
            return None

        if baseline_cfg is None:
            baseline_cfg = BaselineConfig(method="bandpass", lowcut_hz=0.5, highcut_hz=40.0, order=4)

        # FIX: depad and pad to target length BEFORE baseline filter.
        # Previously the filter ran first, crashing on signals shorter than
        # the filter padlen (e.g. 26-sample code15 records at 400 Hz).
        if dataset in ("samitrop", "code15"):
            signal = _depad_trailing_zeros(signal)

        target_len_orig = int(round(10.0 * fs))
        signal = pad_or_trim(signal, target_len_orig)   # guarantees enough samples for filter

        signal = remove_baseline(signal, fs=fs, config=baseline_cfg)

        s500 = resample_signal(signal, original_fs=fs, target_fs=500.0)
        s500 = normalize_per_lead(s500, method="zscore", clip_std=3.0)
        img2d = build_2d_image(s500, target_width=2048, random_crop=make_random_crop, rng=rng)

        s100 = resample_signal(signal, original_fs=fs, target_fs=100.0)
        s100 = normalize_per_lead(s100, method="zscore", clip_std=None)
        s100 = pad_or_trim(s100, 1000).astype(np.float32, copy=False)

        rec_id = record_path.name
        out_2d_dir.mkdir(parents=True, exist_ok=True)
        out_1d_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_2d_dir / f"{rec_id}.npy"
        sig_path = out_1d_dir / f"{rec_id}.npy"
        np.save(img_path, img2d)
        np.save(sig_path, s100)

        return {
            "id": rec_id,
            "dataset": dataset,
            "label_hard": int(label_hard),
            "label_soft": float(label_soft),
            "age": age,
            "sex": sex,
            "wfdb_path": str(record_path),
            "img_path": str(img_path),
            "fm_path": str(sig_path),
        }
    except Exception as e:
        _skip_log.append(f"SKIP {record_path.name}: {type(e).__name__}: {e}")
        return None


def _resolve_code15_record_path(code15_dir: Path, exam_id: str) -> Path:
    flat = code15_dir / str(exam_id)
    sub = code15_dir / "wfdb" / str(exam_id)
    if flat.with_suffix(".hea").exists() or flat.exists():
        return flat
    if sub.with_suffix(".hea").exists() or sub.exists():
        return sub
    return flat


# -------------------- dataset-specific processing -------------------- #

def process_ptbxl(ptbxl_dir: Path, out_2d: Path, out_1d: Path,
                  helper_code_module, subset: float, train_mode: bool):
    meta_csv = ptbxl_dir / "ptbxl_database.csv"
    if not meta_csv.exists():
        meta_csv = ptbxl_dir / "ptbxldatabase.csv"

    df = pd.read_csv(meta_csv)
    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    rng = np.random.default_rng(42)
    out: List[Dict[str, Any]] = []

    _skip_log.clear()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL"):
        fn = str(row.get("filename_hr", "")).replace(".hea", "")
        record_path = ptbxl_dir / fn

        label_hard = int(row.get("chagas", 0)) if "chagas" in row else 0
        label_soft = float(label_hard)

        age = row.get("age", None)
        sex = _normalize_sex_value(row.get("sex", None))

        md = process_single_record(
            record_path=record_path,
            dataset="ptbxl",
            out_2d_dir=out_2d,
            out_1d_dir=out_1d,
            label_hard=label_hard,
            label_soft=label_soft,
            age=age,
            sex=sex,
            helper_code_module=helper_code_module,
            baseline_cfg=BaselineConfig(method="bandpass", lowcut_hz=0.5, highcut_hz=40.0, order=4),
            make_random_crop=train_mode,
            rng=rng,
        )
        if md is not None:
            out.append(md)

    if _skip_log:
        print(f"\n  Skipped {len(_skip_log)} PTB-XL records:")
        for msg in _skip_log:
            print(f"    {msg}")
    return out


def process_samitrop(samitrop_dir: Path, out_2d: Path, out_1d: Path,
                     helper_code_module, subset: float, train_mode: bool):
    meta_csv = samitrop_dir / "exams.csv"
    df = pd.read_csv(meta_csv)
    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    rng = np.random.default_rng(123)
    out: List[Dict[str, Any]] = []

    _skip_log.clear()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SaMi-Trop"):
        exam_id = str(row["exam_id"])
        record_path = samitrop_dir / "wfdb" / exam_id
        if not record_path.with_suffix(".hea").exists():
            record_path = samitrop_dir / exam_id

        label_hard = int(row.get("chagas", 1))
        label_soft = float(label_hard)

        age = row.get("age", None)
        sex = _normalize_sex_value(row.get("is_male", None))

        md = process_single_record(
            record_path=record_path,
            dataset="samitrop",
            out_2d_dir=out_2d,
            out_1d_dir=out_1d,
            label_hard=label_hard,
            label_soft=label_soft,
            age=age,
            sex=sex,
            helper_code_module=helper_code_module,
            baseline_cfg=BaselineConfig(method="bandpass", lowcut_hz=0.5, highcut_hz=40.0, order=4),
            make_random_crop=train_mode,
            rng=rng,
        )
        if md is not None:
            out.append(md)

    if _skip_log:
        print(f"\n  Skipped {len(_skip_log)} SaMi-Trop records:")
        for msg in _skip_log:
            print(f"    {msg}")
    return out


def process_code15(code15_dir: Path, out_2d: Path, out_1d: Path,
                   helper_code_module, subset: float, train_mode: bool):
    label_csv = code15_dir / "code15_chagas_labels.csv"
    if not label_csv.exists():
        label_csv = code15_dir / "code15chagaslabels.csv"
    labels_df = pd.read_csv(label_csv)

    exams_csv = code15_dir / "exams.csv"
    exams_df = pd.read_csv(exams_csv)

    merged = labels_df.merge(
        exams_df[["exam_id", "age", "is_male"]],
        on="exam_id",
        how="left",
    )

    if subset < 1.0:
        merged = merged.sample(frac=subset, random_state=42)

    rng = np.random.default_rng(999)
    out: List[Dict[str, Any]] = []

    _skip_log.clear()
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="CODE-15"):
        exam_id = str(row["exam_id"])
        record_path = _resolve_code15_record_path(code15_dir, exam_id)

        label_hard = int(row.get("chagas", 0))
        label_soft = hard_to_soft_label(label_hard, pos_soft=0.8, neg_soft=0.2)

        age = row.get("age", None)
        sex = _normalize_sex_value(row.get("is_male", None))

        md = process_single_record(
            record_path=record_path,
            dataset="code15",
            out_2d_dir=out_2d,
            out_1d_dir=out_1d,
            label_hard=label_hard,
            label_soft=label_soft,
            age=age,
            sex=sex,
            helper_code_module=helper_code_module,
            baseline_cfg=BaselineConfig(method="bandpass", lowcut_hz=0.5, highcut_hz=40.0, order=4),
            make_random_crop=train_mode,
            rng=rng,
        )
        if md is not None:
            out.append(md)

    if _skip_log:
        print(f"\n  Skipped {len(_skip_log)} CODE-15 records:")
        for msg in _skip_log:
            print(f"    {msg}")
    else:
        print("\n  No CODE-15 records skipped.")
    return out


# -------------------- smart CSV merging -------------------- #

def merge_metadata_csvs(meta_dir: Path, datasets_processed: List[str]) -> pd.DataFrame:
    """
    Intelligently merge existing and new metadata CSVs.
    """
    all_parts = []
    
    # Try to load existing CSVs for datasets NOT being processed
    all_datasets = ["ptbxl", "samitrop", "code15"]
    for ds in all_datasets:
        csv_path = meta_dir / f"{ds}_metadata.csv"
        if csv_path.exists() and ds not in datasets_processed:
            # Keep existing data for this dataset
            df = pd.read_csv(csv_path)
            all_parts.append(df)
            print(f"  OK Loaded existing {ds}: {len(df)} samples")
    
    # Add newly processed datasets
    for ds in datasets_processed:
        csv_path = meta_dir / f"{ds}_metadata.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_parts.append(df)
            print(f"  OK Loaded new {ds}: {len(df)} samples")
    
    if not all_parts:
        return pd.DataFrame()
    
    combined = pd.concat(all_parts, ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    combined = combined.drop_duplicates(subset=['id'], keep='first')
    
    return combined


# -------------------- main -------------------- #

def main():
    p = argparse.ArgumentParser(description="Preprocess ECG datasets with smart merging")
    p.add_argument("--official_wfdb_root", type=str, default="data/official_wfdb")
    p.add_argument("--processed_root", type=str, default="data/processed")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["ptbxl", "samitrop", "code15", "all"],
        default=["all"],
        help="Datasets to process. Use 'all' or specify: ptbxl samitrop code15"
    )
    p.add_argument("--subset", type=float, default=1.0)
    p.add_argument("--train_mode", action="store_true", help="Enable random crop for 2D embedding")
    p.add_argument(
        "--helper_dir",
        type=str,
        default=r"D:\IIT\L6\FYP\ChagaSight\external\official_2025",
        help="Directory containing helper_code.py",
    )
    p.add_argument("--skip_merge", action="store_true", help="Skip merging with existing CSVs")
    args = p.parse_args()

    official = Path(args.official_wfdb_root)
    processed = Path(args.processed_root)

    out_2d = processed / "2d_images"
    out_1d = processed / "1d_signals_100hz"
    meta_dir = processed / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    helper_code_module = _try_import_helper_code(args.helper_dir)

    # Determine which datasets to process
    if "all" in args.datasets:
        datasets_to_process = ["ptbxl", "samitrop", "code15"]
    else:
        datasets_to_process = args.datasets

    print("="*70)
    print(" DATASET PREPROCESSING")
    print("="*70)
    print(f"Datasets to process: {', '.join(datasets_to_process)}")
    print(f"Subset: {args.subset}")
    print(f"Train mode: {args.train_mode}")
    print("="*70 + "\n")

    datasets_processed = []

    # PTB-XL
    if "ptbxl" in datasets_to_process:
        ptbxl_dir = official / "ptbxl"
        if ptbxl_dir.exists():
            print("\n Processing PTB-XL...")
            md = process_ptbxl(
                ptbxl_dir,
                out_2d / "ptbxl",
                out_1d / "ptbxl",
                helper_code_module,
                args.subset,
                args.train_mode,
            )
            pd.DataFrame(md).to_csv(meta_dir / "ptbxl_metadata.csv", index=False)
            datasets_processed.append("ptbxl")
            print(f"OK PTB-XL complete: {len(md)} samples")
        else:
            print(f"! PTB-XL directory not found: {ptbxl_dir}")

    # SaMi-Trop (handles both "samitrop" and "sami_trop")
    if "samitrop" in datasets_to_process:
        samitrop_dir = official / "samitrop"
        if not samitrop_dir.exists():
            samitrop_dir = official / "sami_trop"  # Try alternate name
        
        if samitrop_dir.exists():
            print("\n Processing SaMi-Trop...")
            md = process_samitrop(
                samitrop_dir,
                out_2d / "samitrop",
                out_1d / "samitrop",
                helper_code_module,
                args.subset,
                args.train_mode,
            )
            pd.DataFrame(md).to_csv(meta_dir / "samitrop_metadata.csv", index=False)
            datasets_processed.append("samitrop")
            print(f"OK SaMi-Trop complete: {len(md)} samples")
        else:
            print(f"! SaMi-Trop directory not found. Tried:")
            print(f"  - {official / 'samitrop'}")
            print(f"  - {official / 'sami_trop'}")

    # CODE-15
    if "code15" in datasets_to_process:
        code15_dir = official / "code15"
        if code15_dir.exists():
            print("\n Processing CODE-15...")
            md = process_code15(
                code15_dir,
                out_2d / "code15",
                out_1d / "code15",
                helper_code_module,
                args.subset,
                args.train_mode,
            )
            pd.DataFrame(md).to_csv(meta_dir / "code15_metadata.csv", index=False)
            datasets_processed.append("code15")
            print(f"OK CODE-15 complete: {len(md)} samples")
        else:
            print(f"! CODE-15 directory not found: {code15_dir}")

    # Merge metadata
    print("\n" + "="*70)
    print(" MERGING METADATA")
    print("="*70)
    
    if args.skip_merge:
        print("! Skipping merge (--skip_merge flag)")
        all_df = pd.DataFrame()
        for ds in datasets_processed:
            csv_path = meta_dir / f"{ds}_metadata.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_df = pd.concat([all_df, df], ignore_index=True)
    else:
        all_df = merge_metadata_csvs(meta_dir, datasets_processed)
    
    if len(all_df) > 0:
        all_df.to_csv(meta_dir / "all_data.csv", index=False)
        print(f"\nOK Saved combined: all_data.csv ({len(all_df)} samples)")

    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total samples: {len(all_df)}")
    if len(all_df) > 0:
        print("\nBy dataset:")
        print(all_df["dataset"].value_counts().to_string())
        print(f"\nHard positives: {int(all_df['label_hard'].sum())}")
        print(f"Mean soft label: {float(all_df['label_soft'].mean()):.4f}")
        if "sex" in all_df.columns:
            print("\nSex distribution (1=male, 0=female, NaN=missing):")
            print(all_df["sex"].value_counts(dropna=False).to_string())
    
    print("\n Output locations:")
    print(f"  2D images: {out_2d}")
    print(f"  1D signals: {out_1d}")
    print(f"  Metadata: {meta_dir}")
    print("="*70)


if __name__ == "__main__":
    main()