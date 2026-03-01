import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print("Testing imports...")

# Test model imports
try:
    from src.models.hybrid_model import HybridChagasModel
    print("✓ Model imports OK")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

# Test training imports
try:
    from src.training.dataset import create_dataloaders
    from src.training.trainer import ChagasTrainer
    from src.training.metrics import compute_metrics
    print("✓ Training imports OK")
except Exception as e:
    print(f"✗ Training import failed: {e}")
    sys.exit(1)

# Test model creation
try:
    import torch
    model = HybridChagasModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total:,} parameters")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test CSV
try:
    import pandas as pd
    df = pd.read_csv("data/processed/metadata/combined_5fold.csv")
    print(f"✓ CSV OK: {len(df)} samples, {df['fold'].nunique()} folds")
except Exception as e:
    print(f"✗ CSV failed: {e}")
    sys.exit(1)

print("\n🎉 ALL TESTS PASSED!")