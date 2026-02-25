"""Create stratified 5-fold cross-validation splits (corrected)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_splits() -> pd.DataFrame:
    """Create 5-fold splits stratified by (dataset, label_binary)."""

    metadata_path = Path("data/processed/metadata/all_data.csv")
    df = pd.read_csv(metadata_path)

    print("=" * 60)
    print("CREATING 5-FOLD SPLITS")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(
        f"Positive samples: {df['label'].sum()} "
        f"({df['label'].mean() * 100:.2f}%)"
    )

    # Binary label for stratification
    df["label_binary"] = (df["label"] > 0.5).astype(int)

    # Composite stratification key: dataset + label
    df["strat_key"] = df["dataset"] + "_" + df["label_binary"].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["strat_key"])):
        df.loc[val_idx, "fold"] = fold_idx

    print("\nSplit verification:")
    for fold in range(5):
        fold_df = df[df["fold"] == fold]
        pos = fold_df["label"].sum()
        prev = fold_df["label"].mean() * 100
        print(
            f"  Fold {fold}: {len(fold_df)} samples, "
            f"{pos} positive ({prev:.2f}%)"
        )
        print(f"    Datasets: {fold_df['dataset'].value_counts().to_dict()}")

    output_path = Path("data/processed/metadata/all_data_5fold.csv")
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved 5-fold splits to: {output_path}")
    return df


if __name__ == "__main__":
    create_splits()
