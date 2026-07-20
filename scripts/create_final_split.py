# scripts/create_final_split.py
# Merge dataset_split.csv (teammate's train/val/test assignments) with
# combined_5fold.csv (processed file paths) → final_split.csv
#
# ID normalisation:
#   ptb_xl  : ecg_id=1   → '00001_hr'   (combined_5fold format)
#   code15  : ecg_id=1234 → '1234'
#   samitrop: ecg_id=1234 → '1234'
#
# Output: data/processed/metadata/final_split.csv
#   Same columns as combined_5fold.csv, but with 'split' (train/val/test)
#   instead of 'fold' (0-4).

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = ROOT / "resolved_dataset_split_with_age_filter.csv"
META_CSV  = ROOT / "data" / "processed" / "metadata" / "combined_5fold.csv"
OUT_CSV   = ROOT / "data" / "processed" / "metadata" / "final_split.csv"


def _get_split(path_value: str) -> str | None:
    """Extract 'train', 'val', or 'test' from a split_path string."""
    try:
        parts = Path(path_value).parts
        for part in parts:
            if part in {"train", "val", "test"}:
                return part
    except Exception:
        pass
    return None


def _norm_id(ecg_id: int, dataset: str) -> str:
    """Convert ecg_id to the ID format used in combined_5fold.csv."""
    if dataset == "ptb_xl":
        return f"{int(ecg_id):05d}_hr"
    return str(int(ecg_id))


def _norm_dataset(dataset: str) -> str:
    """Normalise dataset name to match combined_5fold.csv convention."""
    return "ptbxl" if dataset == "ptb_xl" else dataset


def main() -> None:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")

    print(f"Loading split CSV:    {SPLIT_CSV}")
    split_df = pd.read_csv(SPLIT_CSV)
    print(f"  {len(split_df):,} rows | datasets: {split_df['dataset'].value_counts().to_dict()}")

    # Parse split label from path
    split_df["split"] = split_df["split_path"].apply(_get_split)
    n_no_split = split_df["split"].isna().sum()
    if n_no_split:
        print(f"  !  {n_no_split} rows had no parseable split — dropped")
    split_df = split_df[split_df["split"].notna()].copy()

    # Normalise IDs and dataset names
    split_df["id"]      = split_df.apply(lambda r: _norm_id(r["ecg_id"], r["dataset"]), axis=1)
    split_df["dataset"] = split_df["dataset"].apply(_norm_dataset)

    # Build lookup: (dataset, id) → split
    lookup = (
        split_df[["dataset", "id", "split"]]
        .drop_duplicates(subset=["dataset", "id"])  # guard against duplicates
    )

    print(f"\nLoading metadata CSV: {META_CSV}")
    meta_df = pd.read_csv(META_CSV, low_memory=False)
    meta_df["id"] = meta_df["id"].astype(str)
    print(f"  {len(meta_df):,} rows | datasets: {meta_df['dataset'].value_counts().to_dict()}")

    # Merge on (dataset, id) — inner join keeps only assigned records
    merged = meta_df.merge(lookup, on=["dataset", "id"], how="inner")

    # Drop fold column (no longer meaningful)
    merged = merged.drop(columns=["fold"], errors="ignore")

    # -- Report ----------------------------------------------------------------
    print(f"\n{'-'*60}")
    print(f"Merged: {len(merged):,} rows  "
          f"(dropped {len(meta_df) - len(merged):,} records not in split CSV)")
    print()
    for sp in ["train", "val", "test"]:
        sub = merged[merged["split"] == sp]
        pos = int(sub["label_hard"].sum())
        print(f"  {sp:5s}: {len(sub):7,} total  |  {pos:5,} pos ({100*pos/len(sub):.2f}%)")
    print()
    print("Split x dataset x label_hard:")
    print(merged.groupby(["split", "dataset", "label_hard"]).size().to_string())
    print(f"{'-'*60}")

    # -- Save -----------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nOK Saved: {OUT_CSV}")
    print(f"  Columns: {merged.columns.tolist()}")


if __name__ == "__main__":
    main()
