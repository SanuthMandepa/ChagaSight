# src/training/dataset.py - CORRECTED VERSION
"""
ChagaSight Dataset with Soft Labels and Weighted Sampling

FIXED: Custom collate function to handle string fields
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from preprocessing.augmentations import apply_augmentations, DEFAULT_AUGMENTATION_CONFIG
except ImportError:
    print("⚠️  Warning: augmentations module not found. Augmentations disabled.")
    apply_augmentations = None
    DEFAULT_AUGMENTATION_CONFIG = None


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle string fields.
    
    Strings (id, dataset) are kept as lists, not tensors.
    """
    collated = {}
    
    # Handle tensor fields (stack into batch)
    tensor_keys = ['image', 'signal', 'age', 'sex', 'label', 'hard_label']
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle string fields (keep as lists)
    string_keys = ['dataset', 'id']
    for key in string_keys:
        collated[key] = [item[key] for item in batch]
    
    return collated


class ChagasDataset(Dataset):
    """
    Dual-pathway dataset: 2D images + 1D signals + demographics.
    
    Paper: Van Santvliet et al. (2025) + Kim et al. (2025)
    """
    
    def __init__(
        self,
        metadata_csv: str,
        images_dir: str,
        signals_dir: str,
        split: str = 'train',
        fold: int = 0,
        augment: bool = True,
        augmentation_config: Optional[Dict] = None,
        use_soft_labels: bool = True
    ):
        """
        Args:
            metadata_csv: Path to combined_5fold.csv
            images_dir: Path to data/processed/2d_images/
            signals_dir: Path to data/processed/1d_signals_100hz/
            split: 'train' or 'val'
            fold: Fold number (0-4)
            augment: Apply augmentations
            augmentation_config: Custom augmentation config
            use_soft_labels: Use soft labels for CODE-15%
        """
        self.images_dir = Path(images_dir)
        self.signals_dir = Path(signals_dir)
        self.split = split
        self.augment = augment and (split == 'train') and (apply_augmentations is not None)
        self.use_soft_labels = use_soft_labels
        
        if augmentation_config is None and DEFAULT_AUGMENTATION_CONFIG is not None:
            self.augmentation_config = DEFAULT_AUGMENTATION_CONFIG
        else:
            self.augmentation_config = augmentation_config
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        
        # Check for correct column names
        required_cols = ['id', 'dataset', 'label_hard', 'label_soft', 'fold']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # Filter by fold
        if split == 'train':
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        else:  # val
            self.df = df[df['fold'] == fold].reset_index(drop=True)
        
        # Count datasets
        dataset_counts = {}
        for ds in ['ptbxl', 'samitrop', 'code15']:
            count = len(self.df[self.df['dataset'] == ds])
            if count > 0:
                dataset_counts[ds] = count
        
        print(f"✓ Loaded {split} fold {fold}: {len(self.df)} samples")
        print(f"  Datasets: {dataset_counts}")
        print(f"  Positive: {self.df['label_hard'].sum()}, "
              f"Negative: {len(self.df) - self.df['label_hard'].sum()}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Use 'id' column
        record_id = row['id']
        dataset = row['dataset']
        
        # Build proper paths
        image_path = self.images_dir / dataset / f"{record_id}.npy"
        signal_path = self.signals_dir / dataset / f"{record_id}.npy"
        
        # Check if files exist
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal not found: {signal_path}")
        
        # Load 2D image: (3, 24, 2048) uint8
        image = np.load(image_path)
        
        # Load 1D signal: (12, 1000) float32
        signal = np.load(signal_path)
        
        # Apply augmentations to signal (before converting to tensor)
        if self.augment and apply_augmentations is not None:
            signal = apply_augmentations(
                signal,
                self.augmentation_config,
                training=True
            )
        
        # Convert to tensors
        image = torch.from_numpy(image).float()  # (3, 24, 2048)
        signal = torch.from_numpy(signal).float()  # (12, 1000)
        
        # Use label_hard
        hard_label = int(row['label_hard'])
        
        if self.use_soft_labels and dataset == 'code15':
            # Soft labels: Van Santvliet et al. (2025)
            label = 0.8 if hard_label == 1 else 0.2
        else:
            # Hard labels: PTB-XL and SaMi-Trop
            label = float(hard_label)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        # Demographics
        age = row.get('age', 50.0)  # Default to 50 if missing
        if pd.isna(age):
            age = 50.0
        age = float(age) / 100.0  # Convert to centuries [0, 1.2]
        
        sex = row.get('sex', 0.5)  # Default to 0.5 if missing
        if pd.isna(sex):
            sex = 0.5
        sex = float(sex)  # 0=female, 1=male
        
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.float32)
        
        return {
            'image': image,
            'signal': signal,
            'age': age,
            'sex': sex,
            'label': label,
            'hard_label': torch.tensor(hard_label, dtype=torch.long),
            'dataset': dataset,  # String (handled by custom collate)
            'id': record_id  # String (handled by custom collate)
        }
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get sample weights for WeightedRandomSampler.
        
        Paper: Van Santvliet et al. (2025) - 5× oversample positives
        
        Returns:
            weights: Array of sample weights
        """
        labels = self.df['label_hard'].values
        
        # Weight positives 5× more than negatives
        weights = np.ones(len(labels), dtype=np.float32)
        weights[labels == 1] = 5.0
        
        return weights


def create_dataloaders(
    metadata_csv: str,
    images_dir: str,
    signals_dir: str,
    fold: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        metadata_csv: Path to combined_5fold.csv
        images_dir: Path to 2D images
        signals_dir: Path to 1D signals
        fold: Fold number (0-4)
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        use_weighted_sampling: Use weighted sampling for training
        augment_train: Apply augmentations to training data
    
    Returns:
        train_loader, val_loader
    """
    # Training dataset
    train_dataset = ChagasDataset(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        signals_dir=signals_dir,
        split='train',
        fold=fold,
        augment=augment_train,
        use_soft_labels=True
    )
    
    # Validation dataset
    val_dataset = ChagasDataset(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        signals_dir=signals_dir,
        split='val',
        fold=fold,
        augment=False,  # No augmentation for validation
        use_soft_labels=True
    )
    
    # Create samplers
    if use_weighted_sampling:
        # Weighted sampler for training (5× oversample positives)
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False  # Don't shuffle when using sampler
    else:
        train_sampler = None
        shuffle_train = True
    
    # FIXED: Use custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # FIXED: Handle strings
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn  # FIXED: Handle strings
    )
    
    print(f"\n✓ Created dataloaders for fold {fold}:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Weighted sampling: {use_weighted_sampling}")
    print(f"  Augmentation: {augment_train}")
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Test dataset
    train_loader, val_loader = create_dataloaders(
        metadata_csv="data/processed/metadata/combined_5fold.csv",
        images_dir="data/processed/2d_images",
        signals_dir="data/processed/1d_signals_100hz",
        fold=0,
        batch_size=8,
        num_workers=2
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\n✓ Batch shapes:")
    print(f"  image: {batch['image'].shape}")  # (8, 3, 24, 2048)
    print(f"  signal: {batch['signal'].shape}")  # (8, 12, 1000)
    print(f"  age: {batch['age'].shape}")  # (8,)
    print(f"  sex: {batch['sex'].shape}")  # (8,)
    print(f"  label: {batch['label'].shape}")  # (8,)
    print(f"\n✓ String fields:")
    print(f"  dataset (first 3): {batch['dataset'][:3]}")  # List of strings
    print(f"  id (first 3): {batch['id'][:3]}")  # List of strings