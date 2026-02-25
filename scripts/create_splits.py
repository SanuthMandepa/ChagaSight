# create_splits.py
# Create 5-fold splits stratified by (dataset + hard label)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", type=str, default="data/processed/metadata/all_data.csv")
    p.add_argument("--out_dir", type=str, default="data/processed/splits")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.metadata_csv)
    if "label_hard" not in df.columns:
        raise ValueError("Expected label_hard column in metadata CSV.")
    if "dataset" not in df.columns:
        raise ValueError("Expected dataset column in metadata CSV.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = (df["dataset"].astype(str) + "_" + df["label_hard"].astype(int).astype(str)).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    df = df.reset_index(drop=True)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), strat)):
        df.loc[val_idx, "fold"] = fold

    df.to_csv(out_dir / "all_data_with_folds.csv", index=False)

    for fold in range(args.n_splits):
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        train_df.to_csv(out_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(out_dir / f"val_fold{fold}.csv", index=False)

    print("Splits written to:", out_dir)


if __name__ == "__main__":
    main()
