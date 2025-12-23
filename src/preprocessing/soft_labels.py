# src/preprocessing/soft_labels.py
"""
Unified Chagas label assignment.

Implements the dataset-specific label confidence strategy:

    • PTB-XL       → strong negative (0.0)
    • SaMi-Trop    → strong positive (1.0)
    • CODE-15%     → soft labels (0.2 / 0.8)

Soft labeling for CODE-15% accounts for self-reported diagnostic noise
and follows the methodology used in recent challenge-winning approaches.
"""

from typing import Literal, Union, Dict, Any
import warnings
import numpy as np

def get_chagas_label(metadata: Dict[str, Any], dataset: Literal["ptbxl", "sami_trop", "code15"]) -> float:
    """
    Assign Chagas label based on dataset.

    Parameters:
    - metadata: Dict with dataset-specific keys (e.g., 'chagas' for CODE-15%).
    - dataset: One of "ptbxl", "sami_trop", "code15".

    Returns:
    - Float label (0.0 to 1.0).
    """
    if dataset == "ptbxl":
        return 0.0  # All negative per Challenge
    elif dataset == "sami_trop":
        return 1.0  # All positive
    elif dataset == "code15":
        chagas = metadata.get('chagas')
        if chagas is None:
            warnings.warn(f"Missing 'chagas' in metadata for {dataset} - assuming negative")
            return 0.2
        return 0.8 if bool(chagas) else 0.2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def is_confident_label(labels: Union[float, np.ndarray], dataset: Literal["ptbxl", "sami_trop", "code15"]) -> Union[bool, np.ndarray]:
    """
    Check if labels are confident (hard negatives/positives).

    Parameters:
    - labels: Single float or array of labels.
    - dataset: Dataset identifier.

    Returns:
    - Bool or array of bools (True if confident).
    """
    labels = np.asarray(labels)
    
    if dataset == "ptbxl":
        # All PTB-XL labels are confident negatives
        return np.ones_like(labels, dtype=bool)
    elif dataset == "sami_trop":
        # All SaMi-Trop labels are confident positives
        return np.ones_like(labels, dtype=bool)
    elif dataset == "code15":
        # CODE-15% labels near 0 or 1 are more confident
        return (labels < 0.3) | (labels > 0.7)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")