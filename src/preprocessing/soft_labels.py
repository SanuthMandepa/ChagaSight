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


def chagas_label_ptbxl(metadata: Dict[str, Any] = None) -> float:
    """
    PTB-XL is assumed to be Chagas-negative in this project.
    
    Parameters
    ----------
    metadata : dict, optional
        Additional metadata (not used for PTB-XL).
    
    Returns
    -------
    float
        Label 0.0 for Chagas-negative.
    """
    if metadata is not None:
        # Could add checks for specific conditions if needed
        pass
    return 0.0


def chagas_label_sami_trop(metadata: Dict[str, Any] = None) -> float:
    """
    SaMi-Trop cohort is Chagas-positive.
    
    Parameters
    ----------
    metadata : dict, optional
        Additional metadata (not used for SaMi-Trop).
    
    Returns
    -------
    float
        Label 1.0 for Chagas-positive.
    """
    if metadata is not None:
        # Could add checks for specific conditions if needed
        pass
    return 1.0


def chagas_label_code15(
    raw_label: int, 
    metadata: Dict[str, Any] = None,
    soft_positive: float = 0.8,
    soft_negative: float = 0.2
) -> float:
    """
    Soft-label scheme for CODE-15% (self-reported labels).

    Parameters
    ----------
    raw_label : int
        Binary label from CODE-15% metadata:
        1 for positive, 0 for negative.
    metadata : dict, optional
        Additional metadata for more refined labeling.
    soft_positive : float
        Soft label value for positive cases.
    soft_negative : float
        Soft label value for negative cases.

    Returns
    -------
    float
        Soft label.
    """
    if raw_label not in [0, 1]:
        raise ValueError(f"CODE-15% raw_label must be 0 or 1, got {raw_label}")
    
    if metadata is not None:
        # Could incorporate additional metadata for more nuanced labeling
        # For example, quality scores or diagnostic certainty
        pass
    
    if raw_label == 1:
        return soft_positive
    else:
        return soft_negative


def map_dataset_label(
    dataset: Literal["ptbxl", "sami_trop", "code15"], 
    raw_label: Union[int, None] = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> float:
    """
    Unified interface to obtain the Chagas label depending on the dataset.

    Parameters
    ----------
    dataset : {"ptbxl", "sami_trop", "code15"}
    raw_label : int | None
        Original label if available (e.g. for CODE-15%).
    metadata : dict, optional
        Additional metadata for label determination.
    **kwargs : dict
        Additional keyword arguments passed to the specific label function.

    Returns
    -------
    float
        Final label used for Chagas classification.
    """
    if dataset == "ptbxl":
        return chagas_label_ptbxl(metadata=metadata, **kwargs)
    elif dataset == "sami_trop":
        return chagas_label_sami_trop(metadata=metadata, **kwargs)
    elif dataset == "code15":
        if raw_label is None:
            raise ValueError("CODE-15% requires raw_label (0 or 1) to compute soft label.")
        return chagas_label_code15(raw_label, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_label_confidence_mask(
    labels: np.ndarray,
    dataset: str,
    confidence_threshold: float = 0.7
) -> np.ndarray:
    """
    Create a confidence mask based on label values.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of soft labels.
    dataset : str
        Dataset identifier.
    confidence_threshold : float
        Threshold for considering a label confident.
    
    Returns
    -------
    np.ndarray
        Boolean mask indicating confident labels.
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