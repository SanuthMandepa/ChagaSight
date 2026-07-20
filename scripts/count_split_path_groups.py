# scripts/count_split_path_groups.py
# Summary of final_split.csv:
#   - Counts by split/label  (train/0, train/1, val/0 ...)
#   - Per-dataset contribution per split
#   - Missing processed file check (2d_images + 1d_signals_100hz)

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "data" / "processed" / "metadata" / "final_split.csv"
IMG_DIR   = ROOT / "data" / "processed" / "2d_images"
SIG_DIR   = ROOT / "data" / "processed" / "1d_signals_100hz"


def main() -> None:
    df = pd.read_csv(FINAL_CSV, low_memory=False)
    df["id"] = df["id"].astype(str)

    splits  = ["train", "val", "test"]
    labels  = [0, 1]
    datasets = ["code15", "ptbxl", "samitrop"]

    # ── 1. Counts by split / label ────────────────────────────────────────────
    print("Counts by split and label:")
    for sp in splits:
        for lb in labels:
            n = len(df[(df["split"] == sp) & (df["label_hard"] == lb)])
            print(f"  {sp}/{lb}: {n:,}")

    # ── 2. Per-dataset contribution ───────────────────────────────────────────
    print()
    print("Per-dataset contribution:")
    for sp in splits:
        sub = df[df["split"] == sp]
        total = len(sub)
        pos   = int(sub["label_hard"].sum())
        print(f"\n  {sp.upper()}  ({total:,} total | {pos:,} pos | {100*pos/total:.2f}%)")
        for ds in datasets:
            ds_sub = sub[sub["dataset"] == ds]
            if len(ds_sub) == 0:
                continue
            ds_pos = int(ds_sub["label_hard"].sum())
            print(f"    {ds:<10} {len(ds_sub):>7,}  "
                  f"(pos={ds_pos:,}, neg={len(ds_sub)-ds_pos:,})")

    # ── 3. Missing file check ─────────────────────────────────────────────────
    print()
    print("Missing file check:")
    missing_img = []
    missing_sig = []

    for _, row in df.iterrows():
        img = IMG_DIR / row["dataset"] / f"{row['id']}.npy"
        sig = SIG_DIR / row["dataset"] / f"{row['id']}.npy"
        if not img.exists():
            missing_img.append((row["dataset"], row["id"], row["split"]))
        if not sig.exists():
            missing_sig.append((row["dataset"], row["id"], row["split"]))

    if not missing_img and not missing_sig:
        print("  All 2d_images and 1d_signals files present — no missing data.")
    else:
        if missing_img:
            print(f"  MISSING 2d_images : {len(missing_img)}")
            miss_df = pd.DataFrame(missing_img, columns=["dataset", "id", "split"])
            print(miss_df.groupby(["split", "dataset"]).size().to_string())
            print("  First 10:", missing_img[:10])
        if missing_sig:
            print(f"  MISSING 1d_signals: {len(missing_sig)}")
            miss_df = pd.DataFrame(missing_sig, columns=["dataset", "id", "split"])
            print(miss_df.groupby(["split", "dataset"]).size().to_string())
            print("  First 10:", missing_sig[:10])

    # ── 4. Totals ─────────────────────────────────────────────────────────────
    print()
    print("Totals:")
    print(f"  Total records : {len(df):,}")
    print(f"  Total positive: {int(df['label_hard'].sum()):,}")
    print(f"  Total negative: {int((df['label_hard']==0).sum()):,}")


if __name__ == "__main__":
    main()
