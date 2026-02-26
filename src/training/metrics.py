# src/training/metrics.py - CORRECTED VERSION
"""
Metrics for ChagaSight Evaluation

Uses OFFICIAL PhysioNet compute_challenge_score from helper_code.py
Main metric: TPR@5% (True Positive Rate at 5% FPR)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from pathlib import Path

# Import official PhysioNet helper code
# Add path to helper_code.py (adjust if needed)
HELPER_CODE_PATH = Path(__file__).parent.parent.parent / "external" / "official_2025"
if str(HELPER_CODE_PATH) not in sys.path:
    sys.path.insert(0, str(HELPER_CODE_PATH))

try:
    from helper_code import compute_challenge_score, compute_auc
    OFFICIAL_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Official helper_code.py not found. Using approximate TPR@5% calculation.")
    print(f"   Expected location: {HELPER_CODE_PATH}")
    OFFICIAL_METRICS_AVAILABLE = False


def compute_tpr_at_5pct_official(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute TPR@5% using OFFICIAL PhysioNet metric.
    
    This is the PRIMARY METRIC for PhysioNet Challenge 2025.
    
    Uses compute_challenge_score from helper_code.py which:
    - Permutes labels 10,000 times for robustness
    - Computes expected confusion matrix
    - Returns TPR at 5% capacity (top 5% of predictions)
    
    Args:
        labels: Binary labels (0 or 1)
        probs: Predicted probabilities [0, 1]
    
    Returns:
        tpr_5pct: True positive rate at 5% FPR
    """
    if OFFICIAL_METRICS_AVAILABLE:
        # Use official PhysioNet metric
        return compute_challenge_score(
            labels=labels,
            outputs=probs,
            fraction_capacity=0.05,
            num_permutations=10000,
            seed=12345
        )
    else:
        # Fallback to approximate calculation
        return compute_tpr_at_5pct_approximate(labels, probs)


def compute_tpr_at_5pct_approximate(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Approximate TPR@5% calculation (fallback if official not available).
    
    This is LESS ACCURATE than the official metric.
    Use compute_tpr_at_5pct_official() whenever possible.
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(labels, probs)
    
    # Find TPR where FPR <= 0.05
    idx = np.where(fpr <= 0.05)[0]
    if len(idx) == 0:
        return 0.0
    
    return tpr[idx[-1]]


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        labels: Binary labels (0 or 1), shape (N,)
        probs: Predicted probabilities [0, 1], shape (N,)
    
    Returns:
        dict with metrics:
            - tpr_5pct: TPR at 5% FPR (PRIMARY METRIC)
            - auroc: Area under ROC curve
            - auprc: Area under precision-recall curve
            - using_official: bool indicating if official metric was used
    """
    # Ensure numpy arrays
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    # Primary metric: TPR@5% (official PhysioNet metric)
    tpr_5pct = compute_tpr_at_5pct_official(labels, probs)
    
    # Secondary metrics
    if OFFICIAL_METRICS_AVAILABLE:
        # Use official AUROC/AUPRC if available
        auroc, auprc = compute_auc(labels, probs)
    else:
        # Use scikit-learn
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    
    return {
        'tpr_5pct': float(tpr_5pct),  # PRIMARY METRIC
        'auroc': float(auroc),
        'auprc': float(auprc),
        'using_official': OFFICIAL_METRICS_AVAILABLE
    }


# Example usage
if __name__ == "__main__":
    print("Testing metrics with official PhysioNet compute_challenge_score...")
    
    # Create dummy data
    np.random.seed(42)
    labels = np.array([0]*950 + [1]*50)  # 5% positive
    probs = np.random.rand(1000)
    
    # Make positives have higher scores on average
    probs[labels == 1] += 0.3
    probs = np.clip(probs, 0, 1)
    
    # Compute metrics
    metrics = compute_metrics(labels, probs)
    
    print(f"\n✓ Metrics computed:")
    print(f"  TPR@5%: {metrics['tpr_5pct']:.4f} {'(OFFICIAL)' if metrics['using_official'] else '(APPROXIMATE)'}")
    print(f"  AUROC:  {metrics['auroc']:.4f}")
    print(f"  AUPRC:  {metrics['auprc']:.4f}")
    
    if not metrics['using_official']:
        print(f"\n⚠️  Warning: Using approximate metric. Place helper_code.py at:")
        print(f"   {HELPER_CODE_PATH}")