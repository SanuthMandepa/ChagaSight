#!/usr/bin/env python
# team_code.py — ChagaSight: HybridChagasModel for PhysioNet Challenge 2025
"""
ChagaSight Challenge Submission Wrapper

Implements the three required PhysioNet interface functions:
  - train_model(data_folder, model_folder, verbose)
  - load_model(model_folder, verbose)
  - run_model(record, model, verbose)

The model is a HybridChagasModel trained with the full ChagaSight pipeline.
This file wraps the pre-trained ensemble checkpoints for inference.

Architecture:
  - 2D-ViT pathway: processes (3, 24, 2048) ECG contour images
  - 1D-ViT FM pathway: processes (12, 1000) ECG signals + demographics
  - Ensemble of 5 fold models → averaged probability
"""

import numpy as np
import os
import sys
import torch
import json
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent   # ChagaSight/external/official_2025 → ChagaSight/

# Add project source to path
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helper_code import (
    find_records, load_header, get_age, get_sex, get_source,
    load_signals, reorder_signal, save_outputs
)

# ── Preprocessing imports ─────────────────────────────────────────────────────
try:
    from preprocessing.resample import resample_signal, pad_or_trim
    from preprocessing.normalization import normalize_per_lead
    from preprocessing.baseline_removal import remove_baseline
    from preprocessing.image_embedding import build_2d_image
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    print(f'⚠️  Preprocessing not available: {e}')

# ── Model import ──────────────────────────────────────────────────────────────
try:
    from models.hybrid_model import HybridChagasModel
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f'⚠️  HybridChagasModel not available: {e}')

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_FS_1D   = 100    # Hz — Van Santvliet et al.
TARGET_FS_2D   = 100    # Hz — same; images are built at 100Hz
TARGET_LEN     = 1000   # samples at 100Hz = 10 seconds
TARGET_LEN_2D  = 2048   # width in pixels for 2D image
REFERENCE_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

MODEL_CONFIG = dict(
    img_size=(24, 2048), patch_size_2d=(8, 64),
    num_leads=12, seq_len_1d=1000, patch_size_1d=50,
    embed_dim=768, depth=12, num_heads=12,
    use_aol=True, use_demographics=True,
)

DECISION_THRESHOLD = 0.5   # Updated at load time from saved config if available


# ────────────────────────────────────────────────────────────────────────────
# Required function 1: train_model
# ────────────────────────────────────────────────────────────────────────────

def train_model(data_folder, model_folder, verbose):
    """
    Required by PhysioNet interface.

    ChagaSight uses a separate, multi-stage training pipeline:
      1. MAE pretraining (mae_pretraining_2d_COMPLETE.py)
      2. ST-MEM pretraining (stmem_pretraining_1d_COMPLETE.py)
      3. 5-fold fine-tuning (10_train_fold_FINAL_v11.ipynb)

    The checkpoints produced by that pipeline are saved to model_folder
    via save_model(). This function copies them here if already trained,
    or raises an error directing the user to run the pipeline.
    """
    os.makedirs(model_folder, exist_ok=True)
    checkpoint_dir = _PROJECT_ROOT / 'checkpoints'

    # Check if pre-trained checkpoints exist
    fold_paths = [checkpoint_dir / f'fold{i}_best.pt' for i in range(5)]
    available_folds = [p for p in fold_paths if p.exists()]

    if len(available_folds) == 0:
        raise FileNotFoundError(
            'No ChagaSight checkpoints found.\n'
            'Run the full training pipeline:\n'
            '  1. stmem_pretraining_1d_COMPLETE.py  (ST-MEM 1D pretraining)\n'
            '  2. mae_pretraining_2d_COMPLETE.py    (MAE 2D pretraining)\n'
            '  3. 10_train_fold_FINAL_v11.ipynb × 5 (one per fold)\n'
            f'Expected checkpoints at: {checkpoint_dir}'
        )

    if verbose:
        print(f'Found {len(available_folds)}/5 fold checkpoints.')

    # Copy checkpoints to model_folder
    import shutil
    for p in available_folds:
        dest = Path(model_folder) / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
            if verbose:
                print(f'  Copied {p.name} → {model_folder}')

    # Save model config
    config = {
        'model_config': MODEL_CONFIG,
        'n_folds': len(available_folds),
        'threshold': DECISION_THRESHOLD,
    }
    with open(Path(model_folder) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f'Model saved to {model_folder}')


# ────────────────────────────────────────────────────────────────────────────
# Required function 2: load_model
# ────────────────────────────────────────────────────────────────────────────

def load_model(model_folder, verbose):
    """
    Load the ensemble of 5 fold models and return a dict for run_model.
    """
    model_folder = Path(model_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f'Loading ChagaSight ensemble from {model_folder} on {device}')

    # Load config
    config_path = model_folder / 'config.json'
    threshold = DECISION_THRESHOLD
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        threshold = config.get('threshold', DECISION_THRESHOLD)

    # Load all available fold models
    models = []
    for fold in range(5):
        ckpt_path = model_folder / f'fold{fold}_best.pt'
        if not ckpt_path.exists():
            if verbose:
                print(f'  Fold {fold}: checkpoint not found, skipping')
            continue

        m = HybridChagasModel(**MODEL_CONFIG)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.to(device).eval()
        models.append(m)

        if verbose:
            vs = ckpt.get('val_score', None)
            score_str = f'{vs:.4f}' if vs is not None else 'n/a'
            print(f'  Fold {fold}: loaded  (val_score={score_str})')

    if not models:
        raise RuntimeError(f'No fold checkpoints found in {model_folder}')

    if verbose:
        print(f'Loaded {len(models)} fold models')

    return {
        'models': models,
        'device': device,
        'threshold': threshold,
    }


# ────────────────────────────────────────────────────────────────────────────
# Required function 3: run_model
# ────────────────────────────────────────────────────────────────────────────

def run_model(record, model, verbose):
    """
    Run inference on a single record.

    Args:
        record: path to WFDB record (without extension)
        model:  dict from load_model()
        verbose: print debug info

    Returns:
        binary_output:      int (0 or 1)
        probability_output: float in [0, 1]
    """
    models   = model['models']
    device   = model['device']
    threshold = model.get('threshold', DECISION_THRESHOLD)

    # ── Load and preprocess ──────────────────────────────────────────────────
    try:
        signal_1d, image_2d, age_val, sex_val = _load_and_preprocess(record, verbose)
    except Exception as e:
        if verbose:
            print(f'  Preprocessing failed for {record}: {e}')
        return float('nan'), float('nan')

    # ── Inference ────────────────────────────────────────────────────────────
    signal_t = torch.from_numpy(signal_1d).float().unsqueeze(0).to(device)  # (1, 12, 1000)
    image_t  = torch.from_numpy(image_2d).float().unsqueeze(0).to(device)   # (1, 3, 24, 2048)
    age_t    = torch.tensor([age_val],  dtype=torch.float32).to(device)      # (1,)
    sex_t    = torch.tensor([sex_val],  dtype=torch.float32).to(device)      # (1,)

    fold_probs = []
    with torch.no_grad():
        for m in models:
            out  = m(image_t, signal_t, age_t, sex_t)
            prob = float(torch.sigmoid(out['logits']).cpu().item())
            fold_probs.append(prob)

    probability_output = float(np.mean(fold_probs))
    binary_output      = int(probability_output >= threshold)

    if verbose:
        print(f'  {record}: prob={probability_output:.4f}  binary={binary_output}')

    return binary_output, probability_output


# ────────────────────────────────────────────────────────────────────────────
# Internal preprocessing
# ────────────────────────────────────────────────────────────────────────────

def _load_and_preprocess(record, verbose=False):
    """
    Load a WFDB record and return (signal_1d, image_2d, age, sex).

    signal_1d: (12, 1000) float32, z-score normalised, 100 Hz
    image_2d:  (3, 24, 2048) float32 (already normalised 0-1 by PatchEmbed)
    age:       float, age in centuries
    sex:       float, 0.0=female 1.0=male 0.5=unknown
    """
    if not PREPROCESSING_AVAILABLE:
        raise RuntimeError('Preprocessing modules not available')
    if not MODEL_AVAILABLE:
        raise RuntimeError('HybridChagasModel not available')

    # ── Load WFDB ────────────────────────────────────────────────────────────
    header  = load_header(record)
    signal, fields = load_signals(record)

    channels = fields['sig_name']
    fs_orig  = fields['fs']

    # Reorder to standard 12-lead order
    signal = reorder_signal(signal, channels, REFERENCE_LEADS)  # (T, 12)
    signal = signal.T.astype(np.float32)                        # (12, T)

    # ── Demographics ─────────────────────────────────────────────────────────
    age_raw = get_age(header)
    age_val = float(age_raw) / 100.0 if (age_raw is not None and _is_number(age_raw)) else 0.5

    sex_raw = get_sex(header)
    if sex_raw is not None and str(sex_raw).strip().lower().startswith('m'):
        sex_val = 1.0
    elif sex_raw is not None and str(sex_raw).strip().lower().startswith('f'):
        sex_val = 0.0
    else:
        sex_val = 0.5   # unknown

    # ── Baseline removal ─────────────────────────────────────────────────────
    signal = remove_baseline(signal, fs=fs_orig)

    # ── Resample → 100 Hz ────────────────────────────────────────────────────
    signal_100 = resample_signal(signal, original_fs=float(fs_orig), target_fs=float(TARGET_FS_1D))
    signal_100 = pad_or_trim(signal_100, TARGET_LEN)  # (12, 1000)

    # ── 1D signal: z-score normalisation ─────────────────────────────────────
    signal_1d = normalize_per_lead(signal_100, method='zscore', clip_std=3.0)

    # ── 2D image: build from signal (already at 100Hz, 10s) ──────────────────
    # build_2d_image expects (12, T) float, clipped to [-3, 3]
    signal_clipped = np.clip(signal_100, -3.0, 3.0)
    image_2d = build_2d_image(signal_clipped, target_width=TARGET_LEN_2D, random_crop=False)
    # image_2d is (3, 24, 2048) uint8 → convert to float32 for tensor
    image_2d = image_2d.astype(np.float32)

    return signal_1d, image_2d, age_val, sex_val


def _is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# ────────────────────────────────────────────────────────────────────────────
# Optional: save model (called by train_model)
# ────────────────────────────────────────────────────────────────────────────

def save_model(model_folder, model):
    """Save model config. Actual weights already saved by trainer.py."""
    import json
    os.makedirs(model_folder, exist_ok=True)
    config = {
        'model_config': MODEL_CONFIG,
        'n_folds': len(model.get('models', [])) if isinstance(model, dict) else 5,
        'threshold': model.get('threshold', DECISION_THRESHOLD) if isinstance(model, dict) else DECISION_THRESHOLD,
    }
    with open(Path(model_folder) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('team_code.py self-test (interface check only)')
    print(f'  PROJECT_ROOT: {_PROJECT_ROOT}')
    print(f'  PREPROCESSING_AVAILABLE: {PREPROCESSING_AVAILABLE}')
    print(f'  MODEL_AVAILABLE: {MODEL_AVAILABLE}')
    print(f'  Required functions: train_model ✓  load_model ✓  run_model ✓')
    print(f'  Decision threshold: {DECISION_THRESHOLD}')