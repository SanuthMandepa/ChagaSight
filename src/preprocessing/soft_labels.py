"""
Label utilities for Chagas classification across PTB-XL, CODE-15%, and SaMi-Trop.
"""

from typing import Literal


def chagas_label_ptbxl() -> float:
    """
    PTB-XL is assumed to be Chagas-negative in this project.
    """
    return 0.0


def chagas_label_sami_trop() -> float:
    """
    SaMi-Trop cohort is Chagas-positive.
    """
    return 1.0


def chagas_label_code15(raw_label: int) -> float:
    """
    Soft-label scheme for CODE-15% (self-reported labels).

    Parameters
    ----------
    raw_label : int
        Binary label from CODE-15% metadata:
        1 for positive, 0 for negative.

    Returns
    -------
    float
        Soft label, 0.8 for positives, 0.2 for negatives.
    """
    return 0.8 if raw_label == 1 else 0.2


def map_dataset_label(dataset: Literal["ptbxl", "sami_trop", "code15"], raw_label: int | None) -> float:
    """
    Unified interface to obtain the Chagas label depending on the dataset.

    Parameters
    ----------
    dataset : {"ptbxl", "sami_trop", "code15"}
    raw_label : int | None
        Original label if available (e.g. for CODE-15%).

    Returns
    -------
    float
        Final label used for Chagas classification.
    """
    if dataset == "ptbxl":
        return chagas_label_ptbxl()
    elif dataset == "sami_trop":
        return chagas_label_sami_trop()
    elif dataset == "code15":
        if raw_label is None:
            raise ValueError("CODE-15% requires raw_label (0 or 1) to compute soft label.")
        return chagas_label_code15(raw_label)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
