# scripts/build_all_data.py
# End-to-end preprocessing for PTB-XL, SaMi-Trop, CODE-15 in WFDB format.
# Produces:
#   - 2D images: (3,24,2048) uint8 as .npy
#   - 1D FM signals: (12,1000) float32 as .npy
#   - metadata CSV(s) with label_hard + label_soft and binary sex (1=male, 0=female)

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# ---------------------------------------------------------------------
# Add <project_root>/src to sys.path so we can import src.preprocessing.*
# ---------------------------------------------------------------------
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
    Accepts: bool, int, float, str.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None

    # Booleans directly: True -> 1, False -> 0
    if isinstance(raw, bool):
        return 1 if raw else 0

    # Numeric types
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) == 1:
            return 1
        if int(raw) == 0:
            return 0

    # Strings
    s = str(raw).strip().lower()
    if s in {"m", "male", "1", "true", "t", "yes", "y"}:
        return 1
    if s in {"f", "female", "0", "false", "f", "no", "n"}:
        return 0

    return None


def _try_import_helper_code(helper_dir: Optional[str]):
    """
    If helper_dir is provided, it must be the DIRECTORY containing helper_code.py,
    e.g. r"D:\\IIT\\L6\\FYP\\ChagaSight\\external\\official_2025".
    """
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
    """
    Returns reordered signal in STANDARD_12 order as (12, T).
    If helper_code is unavailable, we assume signal is already in the correct order.
    """
    if helper_code_module is None:
        return psignal.T.astype(np.float32, copy=False)

    try:
        reordered = helper_code_module.reordersignal(psignal, sig_names, STANDARD_12)  # (T, 12)
        return reordered.T.astype(np.float32, copy=False)
    except Exception:
        return psignal.T.astype(np.float32, copy=False)


def _depad_trailing_zeros(signal: np.ndarray) -> np.ndarray:
    """
    CODE-15 and SaMi-Trop can have trailing zero padding. We remove it by finding
    the last time index where any lead is non-zero.
    """
    if signal.ndim != 2:
        raise ValueError("signal must be (L,T)")
    non_zero_mask = np.any(signal != 0.0, axis=0)
    if not np.any(non_zero_mask):
        return signal
    last = int(np.where(non_zero_mask)[0][-1])
    return signal[:, : last + 1]


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
        psignal = record.p_signal  # (T, C)
        sig_names = list(getattr(record, "sig_name", []))

        signal = _reorder_to_standard(psignal, sig_names, helper_code_module)  # (12, T)
        fs = float(record.fs)

        if signal.shape[0] != 12:
            return None

        # Baseline removal
        if baseline_cfg is None:
            baseline_cfg = BaselineConfig(method="bandpass", lowcut_hz=0.5, highcut_hz=40.0, order=4)
        signal = remove_baseline(signal, fs=fs, config=baseline_cfg)  # (12,T)

        # Depad for datasets known to have trailing zeros
        if dataset in ("samitrop", "code15"):
            signal = _depad_trailing_zeros(signal)

        # Pad/trim to exactly 10 seconds at original fs
        target_len_orig = int(round(10.0 * fs))
        signal = pad_or_trim(signal, target_len_orig)  # (12, target_len_orig)

        # 2D branch: resample to 500 Hz, normalize (clip), build image
        s500 = resample_signal(signal, original_fs=fs, target_fs=500.0)  # (12, ~5000)
        s500 = normalize_per_lead(s500, method="zscore", clip_std=3.0)
        img2d = build_2d_image(s500, target_width=2048, random_crop=make_random_crop, rng=rng)

        # 1D FM branch: resample to 100 Hz, normalize (no clip), pad/trim to 1000
        s100 = resample_signal(signal, original_fs=fs, target_fs=100.0)  # (12, ~1000)
        s100 = normalize_per_lead(s100, method="zscore", clip_std=None)
        s100 = pad_or_trim(s100, 1000).astype(np.float32, copy=False)

        # Save
        rec_id = record_path.name  # works for "3108556" etc.
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
    except Exception:
        return None


def _resolve_code15_record_path(code15_dir: Path, exam_id: str) -> Path:
    """
    CODE-15 WFDB files may be flat under code15_dir (no wfdb/ subfolder).
    """
    flat = code15_dir / str(exam_id)
    sub = code15_dir / "wfdb" / str(exam_id)
    if flat.with_suffix(".hea").exists() or flat.exists():
        return flat
    if sub.with_suffix(".hea").exists() or sub.exists():
        return sub
    # default to flat (so failures are consistent and obvious)
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL"):
        fn = str(row.get("filename_hr", "")).replace(".hea", "")
        record_path = ptbxl_dir / fn

        # Chagas hard label: PTB-XL is assumed negative (0)
        label_hard = int(row.get("chagas", 0)) if "chagas" in row else 0
        label_soft = float(label_hard)

        age = row.get("age", None)
        raw_sex = row.get("sex", None)
        sex = _normalize_sex_value(raw_sex)

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

    return out


def process_samitrop(samitrop_dir: Path, out_2d: Path, out_1d: Path,
                     helper_code_module, subset: float, train_mode: bool):
    meta_csv = samitrop_dir / "exams.csv"
    df = pd.read_csv(meta_csv)
    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    rng = np.random.default_rng(123)
    out: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SaMi-Trop"):
        exam_id = str(row["exam_id"])
        # SaMi-Trop often stored under samitrop_dir/wfdb/<exam_id>
        record_path = samitrop_dir / "wfdb" / exam_id
        if not record_path.with_suffix(".hea").exists():
            record_path = samitrop_dir / exam_id

        label_hard = int(row.get("chagas", 1))
        label_soft = float(label_hard)  # keep hard for SaMi-Trop

        age = row.get("age", None)
        # Use is_male from exams.csv and convert to binary sex
        raw_is_male = row.get("is_male", None)
        sex = _normalize_sex_value(raw_is_male)

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

    return out


def process_code15(code15_dir: Path, out_2d: Path, out_1d: Path,
                   helper_code_module, subset: float, train_mode: bool):
    # Accept either filename
    meta_csv = code15_dir / "code15_chagas_labels.csv"
    if not meta_csv.exists():
        meta_csv = code15_dir / "code15chagaslabels.csv"

    df = pd.read_csv(meta_csv)
    if subset < 1.0:
        df = df.sample(frac=subset, random_state=42)

    rng = np.random.default_rng(999)
    out: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CODE-15"):
        exam_id = str(row["exam_id"])
        record_path = _resolve_code15_record_path(code15_dir, exam_id)

        label_hard = int(row.get("chagas", 0))
        label_soft = hard_to_soft_label(label_hard, pos_soft=0.8, neg_soft=0.2)

        age = row.get("age", None)
        raw_sex = row.get("sex", None)
        sex = _normalize_sex_value(raw_sex)

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

    return out


# -------------------- main -------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official_wfdb_root", type=str, default="data/official_wfdb")
    p.add_argument("--processed_root", type=str, default="data/processed")
    p.add_argument("--subset", type=float, default=1.0)
    p.add_argument("--train_mode", action="store_true", help="Enable random crop for 2D embedding")
    p.add_argument(
        "--helper_dir",
        type=str,
        default=r"D:\IIT\L6\FYP\ChagaSight\external\official_2025",
        help="Directory containing helper_code.py",
    )
    args = p.parse_args()

    official = Path(args.official_wfdb_root)
    processed = Path(args.processed_root)

    out_2d = processed / "2d_images"
    out_1d = processed / "1d_signals_100hz"
    meta_dir = processed / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    helper_code_module = _try_import_helper_code(args.helper_dir)

    all_md: List[Dict[str, Any]] = []

    if (official / "ptbxl").exists():
        md = process_ptbxl(
            official / "ptbxl",
            out_2d / "ptbxl",
            out_1d / "ptbxl",
            helper_code_module,
            args.subset,
            args.train_mode,
        )
        pd.DataFrame(md).to_csv(meta_dir / "ptbxl_metadata.csv", index=False)
        all_md.extend(md)

    if (official / "samitrop").exists():
        md = process_samitrop(
            official / "samitrop",
            out_2d / "samitrop",
            out_1d / "samitrop",
            helper_code_module,
            args.subset,
            args.train_mode,
        )
        pd.DataFrame(md).to_csv(meta_dir / "samitrop_metadata.csv", index=False)
        all_md.extend(md)

    if (official / "code15").exists():
        md = process_code15(
            official / "code15",
            out_2d / "code15",
            out_1d / "code15",
            helper_code_module,
            args.subset,
            args.train_mode,
        )
        pd.DataFrame(md).to_csv(meta_dir / "code15_metadata.csv", index=False)
        all_md.extend(md)

    all_df = pd.DataFrame(all_md)
    all_df.to_csv(meta_dir / "all_data.csv", index=False)

    print("Preprocessing complete.")
    print(f"Total samples: {len(all_df)}")
    if len(all_df) > 0:
        print("By dataset:\n", all_df["dataset"].value_counts())
        print("Hard positives:", int(all_df["label_hard"].sum()))
        print("Mean soft label:", float(all_df["label_soft"].mean()))
        if "sex" in all_df.columns:
            print("Sex value counts (1=male,0=female,NaN=missing):")
            print(all_df["sex"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
