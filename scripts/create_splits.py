# create_splits.py - CORRECTED VERSION
# Create 5-fold splits stratified by (dataset + hard label)
# SAVES combined_5fold.csv to data/processed/metadata/ (not splits/)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", type=str, default="data/processed/metadata/all_data.csv")
    p.add_argument("--out_splits_dir", type=str, default="data/processed/splits")
    p.add_argument("--out_metadata_dir", type=str, default="data/processed/metadata")  # NEW
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.metadata_csv)
    
    print(f"OK Loaded {len(df)} samples from {args.metadata_csv}")
    print(f"  Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"  Positives: {df['label_hard'].sum()}")
    
    # Check for required columns
    if "label_hard" not in df.columns:
        raise ValueError("Expected label_hard column in metadata CSV.")
    if "dataset" not in df.columns:
        raise ValueError("Expected dataset column in metadata CSV.")

    # Create output directories
    splits_dir = Path(args.out_splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_dir = Path(args.out_metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by (dataset + label_hard)
    strat = (df["dataset"].astype(str) + "_" + df["label_hard"].astype(int).astype(str)).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    df = df.reset_index(drop=True)
    df["fold"] = -1

    # Assign fold numbers
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), strat)):
        df.loc[val_idx, "fold"] = fold

    # CORRECTED: Save combined_5fold.csv to metadata/ directory
    combined_csv_path = metadata_dir / "combined_5fold.csv"
    df.to_csv(combined_csv_path, index=False)
    print(f"\nOK Saved combined CSV: {combined_csv_path}")

    # Also save to splits/ for backwards compatibility
    df.to_csv(splits_dir / "all_data_with_folds.csv", index=False)
    print(f"OK Saved (backup): {splits_dir / 'all_data_with_folds.csv'}")

    # Create per-fold train/val CSVs
    for fold in range(args.n_splits):
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        
        train_df.to_csv(splits_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(splits_dir / f"val_fold{fold}.csv", index=False)
        
        print(f"  Fold {fold}: train={len(train_df)}, val={len(val_df)}")

    print(f"\nOK Splits written to: {splits_dir}")
    print(f"OK Combined CSV at: {combined_csv_path}")
    
    # Print fold distribution
    print("\n Fold distribution:")
    for fold in range(args.n_splits):
        fold_data = df[df['fold'] == fold]
        pos = fold_data['label_hard'].sum()
        neg = len(fold_data) - pos
        print(f"  Fold {fold}: {len(fold_data)} samples ({pos} pos, {neg} neg)")


if __name__ == "__main__":
    main()