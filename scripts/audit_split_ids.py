# scripts/audit_split_ids.py
# Reconcile IDs between dataset_split.csv (teammate) and final_split.csv (ours)
# Identifies which IDs are missing on either side, per dataset.

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
SPLIT_CSV  = ROOT / "resolved_dataset_split_with_age_filter.csv"
FINAL_CSV  = ROOT / "data" / "processed" / "metadata" / "final_split.csv"


def norm_id(ecg_id: int, dataset: str) -> str:
    """Same normalisation used in create_final_split.py."""
    if dataset == "ptb_xl":
        return f"{int(ecg_id):05d}_hr"
    return str(int(ecg_id))


def norm_dataset(dataset: str) -> str:
    return "ptbxl" if dataset == "ptb_xl" else dataset


def get_split(path_value: str) -> str | None:
    for part in Path(path_value).parts:
        if part in {"train", "val", "test"}:
            return part
    return None


def main() -> None:
    # ── Load teammate split CSV ───────────────────────────────────────────────
    split_df = pd.read_csv(SPLIT_CSV)
    split_df["split"]       = split_df["split_path"].apply(get_split)
    split_df["id_norm"]     = split_df.apply(lambda r: norm_id(r["ecg_id"], r["dataset"]), axis=1)
    split_df["dataset_norm"]= split_df["dataset"].apply(norm_dataset)

    # ── Load our final_split.csv ──────────────────────────────────────────────
    final_df = pd.read_csv(FINAL_CSV, low_memory=False)
    final_df["id"] = final_df["id"].astype(str)

    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  dataset_split.csv : {len(split_df):,} rows")
    print(f"  final_split.csv   : {len(final_df):,} rows")
    print()

    datasets = [("ptbxl", "ptb_xl"), ("code15", "code15"), ("samitrop", "samitrop")]

    for ds_norm, ds_raw in datasets:
        split_ids = set(
            split_df[split_df["dataset_norm"] == ds_norm]["id_norm"]
        )
        final_ids = set(
            final_df[final_df["dataset"] == ds_norm]["id"]
        )

        only_in_split = split_ids - final_ids   # teammate has, we don't
        only_in_final = final_ids - split_ids   # we have, teammate doesn't
        matched       = split_ids & final_ids

        print("-" * 65)
        print(f"  DATASET: {ds_norm}")
        print(f"    teammate CSV  : {len(split_ids):,} IDs")
        print(f"    final_split   : {len(final_ids):,} IDs")
        print(f"    matched       : {len(matched):,} IDs")
        print(f"    only in teammate (not in ours): {len(only_in_split)}")
        print(f"    only in ours (not in teammate): {len(only_in_final)}")

        # Show split breakdown for matched vs only-in-final
        if only_in_split:
            print(f"\n    IDs in teammate CSV but MISSING from final_split:")
            # Show with their split assignment
            rows = split_df[
                (split_df["dataset_norm"] == ds_norm) &
                (split_df["id_norm"].isin(only_in_split))
            ][["id_norm", "split", "chagas"]].sort_values("split")
            print(rows.groupby(["split", "chagas"]).size().to_string())
            if len(only_in_split) <= 20:
                print("    IDs:", sorted(only_in_split))

        if only_in_final:
            print(f"\n    IDs in final_split but NOT in teammate CSV:")
            rows = final_df[
                (final_df["dataset"] == ds_norm) &
                (final_df["id"].isin(only_in_final))
            ][["id", "label_hard", "split"]].sort_values("split")
            print(rows.groupby(["split", "label_hard"]).size().to_string())
            if len(only_in_final) <= 20:
                print("    IDs:", sorted(only_in_final))

        print()

    # ── Duplicate check inside teammate CSV ───────────────────────────────────
    print("=" * 65)
    print("DUPLICATE IDs IN TEAMMATE CSV (per dataset)")
    print("=" * 65)
    for ds_norm, ds_raw in datasets:
        sub = split_df[split_df["dataset_norm"] == ds_norm]
        dups = sub[sub.duplicated("id_norm", keep=False)]
        print(f"  {ds_norm}: {len(dups)} duplicate rows "
              f"({dups['id_norm'].nunique()} unique IDs duplicated)")
        if len(dups) > 0:
            print(dups[["id_norm","split","chagas"]].sort_values("id_norm").to_string())

    # ── Val / test cross-check ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("VAL / TEST ID SETS - FINAL SPLIT BREAKDOWN")
    print("=" * 65)
    for sp in ["val", "test"]:
        sub = final_df[final_df["split"] == sp]
        print(f"\n  {sp.upper()} ({len(sub):,} total, {int(sub['label_hard'].sum())} pos):")
        print(sub.groupby(["dataset", "label_hard"]).size().to_string())


if __name__ == "__main__":
    main()
