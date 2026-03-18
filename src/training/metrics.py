# src/training/metrics.py  ── v6 DEFINITIVE
"""
Metrics for ChagaSight Evaluation

PRIMARY METRIC: TPR@5% via official PhysioNet compute_challenge_score

CONFIRMED PATH:
  helper_code.py is at: external/official_2025/helper_code.py
  metrics.py is at:     src/training/metrics.py
  
  Path(__file__).parent.parent.parent = project_root (ChagaSight/)
  / "external" / "official_2025" / "helper_code.py"  ← CONFIRMED CORRECT

v6 CHANGE: compute_metrics() now accepts num_permutations parameter.
  Trainer uses 1000 for fast mid-training checks (~0.1s)
  and 10000 for end-of-phase official scores (~1.5s).

WHAT THE METRIC ACTUALLY MEANS:
  capacity = int(0.05 × N) = 831 patients on val_fold0 (N=16,626)
  "If a hospital can only screen 5% of patients, what fraction of
   Chagas cases does the model correctly identify to screen?"
  
  val_fold0: 565 true positives, 16,061 negatives
  Random:    TPR@5% ≈ 0.05  (28 of 565 found in top 831)
  Phase 1:   TPR@5% ≈ 0.138  (78 of 565 found — 2.8× better than random) ✓
  Target:    TPR@5% ≥ 0.42  (237 of 565 found)
  Top team:  TPR@5% = 0.445 (251 of 565 found)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import sys
import warnings
from pathlib import Path

# ── Find helper_code.py ──────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # src/training/../../.. = ChagaSight/

_SEARCH_PATHS = [
    _PROJECT_ROOT / "external" / "official_2025",   # PRIMARY — confirmed
    _PROJECT_ROOT,                                   # project root fallback
    Path.cwd() / "external" / "official_2025",      # notebook cwd fallback
    Path.cwd(),
]

OFFICIAL_METRICS_AVAILABLE = False
_HELPER_CODE_PATH = None

for _path in _SEARCH_PATHS:
    _candidate = _path / "helper_code.py"
    if _candidate.exists():
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
        try:
            from helper_code import compute_challenge_score, compute_auc
            OFFICIAL_METRICS_AVAILABLE = True
            _HELPER_CODE_PATH = _candidate
            break
        except ImportError:
            continue

if OFFICIAL_METRICS_AVAILABLE:
    print(f"✓ Official PhysioNet helper_code loaded from:")
    print(f"  {_HELPER_CODE_PATH}")
else:
    print("  helper_code.py not found — using approximate TPR@5% (less accurate).")
    print(f"   Expected at: {_SEARCH_PATHS[0] / 'helper_code.py'}")


# ── Challenge score ───────────────────────────────────────────────────────────

def compute_tpr_at_5pct(labels: np.ndarray, probs: np.ndarray,
                         num_permutations: int = 10000) -> float:
    """
    Compute official PhysioNet TPR@5% challenge score.

    Args:
        labels:           (N,) binary 0/1 labels
        probs:            (N,) predicted probabilities [0, 1]
        num_permutations: 10000 for official score, 1000 for fast estimate

    Returns: tpr float in [0, 1]
    """
    if OFFICIAL_METRICS_AVAILABLE:
        return float(compute_challenge_score(
            labels=np.asarray(labels, dtype=np.float64),
            outputs=np.asarray(probs,  dtype=np.float64),
            fraction_capacity=0.05,
            num_permutations=num_permutations,
            seed=12345,
        ))
    else:
        return _tpr_at_5pct_approx(labels, probs)


def _tpr_at_5pct_approx(labels: np.ndarray, probs: np.ndarray) -> float:
    """Approximate fallback: TPR at 5% FPR on the ROC curve (different metric)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.where(fpr <= 0.05)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0


# ── Main metrics function ─────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    num_permutations: int = 10000) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        labels:           (N,) binary 0/1 hard labels
        probs:            (N,) predicted probabilities [0, 1]
        num_permutations: permutations for TPR@5%.
                          10000 = official (default, for end-of-phase).
                          1000  = fast estimate for mid-training checks.

    Returns dict:
        tpr_5pct         : PRIMARY METRIC — TPR at 5% capacity
        auroc            : Area under ROC curve
        auprc            : Area under PR curve
        using_official   : True if official helper_code used
        num_permutations : permutations used (for display)
        n_pos, n_total   : dataset statistics
    """
    labels = np.asarray(labels, dtype=np.float64)
    probs  = np.asarray(probs,  dtype=np.float64)
    n_total = len(labels)
    n_pos   = int(labels.sum())

    if len(np.unique(labels)) < 2 or n_pos == 0:
        return {
            'tpr_5pct': 0.0, 'auroc': 0.5, 'auprc': 0.0,
            'using_official': OFFICIAL_METRICS_AVAILABLE,
            'num_permutations': num_permutations,
            'n_pos': n_pos, 'n_total': n_total,
        }

    tpr_5pct = compute_tpr_at_5pct(labels, probs, num_permutations=num_permutations)

    if OFFICIAL_METRICS_AVAILABLE:
        try:
            auroc, auprc = compute_auc(labels, probs)
        except Exception:
            auroc = float(roc_auc_score(labels, probs))
            auprc = float(average_precision_score(labels, probs))
    else:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))

    return {
        'tpr_5pct':         float(tpr_5pct),
        'auroc':            float(auroc),
        'auprc':            float(auprc),
        'using_official':   OFFICIAL_METRICS_AVAILABLE,
        'num_permutations': num_permutations,
        'n_pos':            n_pos,
        'n_total':          n_total,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("metrics.py self-test")
    print("=" * 60)
    np.random.seed(42)
    N, n_pos = 16626, 565
    labels = np.array([1]*n_pos + [0]*(N-n_pos))
    probs  = np.clip(np.random.rand(N) + 0.4 * (labels == 1), 0, 1)

    # Fast (1000 perms)
    m_fast = compute_metrics(labels, probs, num_permutations=1000)
    print(f"\nFast (1000 perms):  TPR@5%={m_fast['tpr_5pct']:.4f}  AUROC={m_fast['auroc']:.4f}")

    # Full (10000 perms)
    m_full = compute_metrics(labels, probs, num_permutations=10000)
    print(f"Full (10000 perms): TPR@5%={m_full['tpr_5pct']:.4f}  AUROC={m_full['auroc']:.4f}")
    print(f"Method: {'OFFICIAL' if m_full['using_official'] else 'APPROXIMATE'}")
    print(f"\nScore benchmarks (N=16626, 3.4% positive):")
    print(f"  Random:   ~0.050")
    print(f"  Phase 1:   0.138 (your current)")
    print(f"  Phase 2:  ~0.280-0.350 (expected without pretraining)")
    print(f"  Target:    0.420")
    print(f"  Top team:  0.445")