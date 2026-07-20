# src/training/metrics.py  ── v7
"""
Metrics for ChagaSight Evaluation

PRIMARY METRIC: TPR@5% via official PhysioNet compute_challenge_score

v7 CHANGE: compute_metrics() now also returns Youden-optimal threshold and
  binary metrics (precision, recall, accuracy) computed at that threshold.
  F1 is excluded per supervisor guidance.
  Trainer uses these for TensorBoard logging and history.

v6 CHANGE: compute_metrics() now accepts num_permutations parameter.
  Trainer uses 1000 for fast mid-training checks (~0.1s)
  and 10000 for end-of-phase official scores (~1.5s).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_score, recall_score, accuracy_score, f1_score,
)
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

def _full_metrics_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Compute all binary classification metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec  = float(recall_score(labels, preds, zero_division=0))
    f1   = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2   = 5.0 * prec * rec / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0
    acc  = float(accuracy_score(labels, preds))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'prec': prec, 'rec': rec, 'f1': f1, 'f2': f2,
        'acc': acc, 'spec': spec, 'npv': npv,
    }


def threshold_youden(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Youden's J: argmax(TPR - FPR) on the ROC curve.
    Maximises sensitivity + specificity simultaneously.
    Standard choice for imbalanced medical screening.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr_c, tpr_c, thresholds_c = roc_curve(labels, probs)
    J = tpr_c - fpr_c
    return float(thresholds_c[int(np.argmax(J))])


def threshold_min_precision(labels: np.ndarray, probs: np.ndarray,
                             min_precision: float = 0.30) -> float:
    """
    Constraint: precision >= min_precision.
    Among all thresholds meeting that constraint, returns the lowest one
    (= maximum recall while keeping precision acceptable).
    Falls back to Youden if constraint cannot be satisfied.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr_c, tpr_c, thresholds_c = roc_curve(labels, probs)
    # Walk from high threshold (high precision, low recall) downward
    for i in range(len(thresholds_c) - 1, -1, -1):
        thr = float(thresholds_c[i])
        preds = (probs >= thr).astype(int)
        p = float(precision_score(labels, preds, zero_division=0))
        if p >= min_precision:
            return thr
    # Constraint cannot be satisfied — fall back
    return threshold_youden(labels, probs)


def threshold_min_recall(labels: np.ndarray, probs: np.ndarray,
                          min_recall: float = 0.85) -> float:
    """
    Constraint: recall >= min_recall.
    Among all thresholds meeting that constraint, returns the highest one
    (= maximum precision while keeping recall acceptable).
    Falls back to Youden if constraint cannot be satisfied.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpr_c, tpr_c, thresholds_c = roc_curve(labels, probs)
    # Walk from low threshold (high recall) upward
    for i in range(len(thresholds_c)):
        thr = float(thresholds_c[i])
        preds = (probs >= thr).astype(int)
        r = float(recall_score(labels, preds, zero_division=0))
        if r >= min_recall:
            return thr
    # Constraint cannot be satisfied — fall back
    return threshold_youden(labels, probs)


def threshold_max_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Max-F1: scan all ROC-curve thresholds and return the one that maximises F1.
    F1 = 2*P*R / (P+R) — balances precision and recall simultaneously.
    Supervisor-requested: threshold selected using recall (via F1).
    Falls back to Youden if no valid threshold found.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, thresholds_c = roc_curve(labels, probs)
    best_f1 = -1.0
    best_thr = threshold_youden(labels, probs)  # safe fallback
    for thr in thresholds_c:
        preds = (probs >= thr).astype(int)
        p = float(precision_score(labels, preds, zero_division=0))
        r = float(recall_score(labels, preds, zero_division=0))
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
    return best_thr


def threshold_t05(labels: np.ndarray, probs: np.ndarray) -> float:
    """Fixed threshold at 0.5 — standard sigmoid cutoff."""
    return 0.5


# Supported threshold strategies
THRESHOLD_STRATEGIES = {
    'youden':        threshold_youden,               # argmax(TPR - FPR)
    'max_f1':        threshold_max_f1,               # argmax F1
    'min_precision': threshold_min_precision,        # max recall s.t. precision >= 0.30
    'min_recall':    threshold_min_recall,           # max precision s.t. recall >= 0.85
    't05':           threshold_t05,                  # fixed sigmoid cutoff 0.5
}


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    num_permutations: int = 10000,
                    threshold_strategy: str = 'youden',
                    threshold_kwargs: dict = None,
                    rec_min_recall: float = 0.99,
                    recp_min_precision: float = 0.30) -> dict:
    """
    Compute all evaluation metrics including 4 named threshold strategies.

    Named strategies (Dewmika format, always computed):
        rec   - highest recall  (threshold_min_recall >= rec_min_recall)
        recp  - recall+precision (threshold_min_precision >= recp_min_precision)
        f1    - maximise F1     (threshold_max_f1)
        t05   - fixed 0.5 cutoff
    Each strategy returns: thr, tp, fp, tn, fn, acc, prec, rec, spec, npv, f1, f2
    """
    labels = np.asarray(labels, dtype=np.float64)
    probs  = np.asarray(probs,  dtype=np.float64)
    n_total = len(labels)
    n_pos   = int(labels.sum())

    def _zero_strat(s: str, thr: float = float('nan')) -> dict:
        return {
            f'{s}_thr': thr,  f'{s}_tp': 0,    f'{s}_fp': 0,
            f'{s}_tn': 0,     f'{s}_fn': 0,    f'{s}_acc': 0.0,
            f'{s}_prec': 0.0, f'{s}_rec': 0.0, f'{s}_spec': 0.0,
            f'{s}_npv': 0.0,  f'{s}_f1': 0.0,  f'{s}_f2': 0.0,
        }

    if len(np.unique(labels)) < 2 or n_pos == 0:
        d = {
            'tpr_5pct': 0.0, 'auroc': 0.5, 'auprc': 0.0,
            'threshold': float('nan'), 'threshold_strategy': threshold_strategy,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'threshold_youden': float('nan'),
            'precision_youden': 0.0, 'recall_youden': 0.0, 'f1_youden': 0.0,
            'accuracy_youden': 0.0,
            'tp_youden': 0, 'fp_youden': 0, 'fn_youden': 0, 'tn_youden': 0,
            'precision_t05': 0.0, 'recall_t05': 0.0, 'f1_t05': 0.0, 'accuracy_t05': 0.0,
            'tp_t05': 0, 'fp_t05': 0, 'fn_t05': 0, 'tn_t05': 0,
            'using_official': OFFICIAL_METRICS_AVAILABLE,
            'num_permutations': num_permutations,
            'n_pos': n_pos, 'n_total': n_total,
        }
        for s in ('rec', 'recp', 'f1', 't05'):
            d.update(_zero_strat(s, thr=0.5 if s == 't05' else float('nan')))
        return d

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

    # Primary threshold (strategy-selected, backward compat)
    thr_fn    = THRESHOLD_STRATEGIES.get(threshold_strategy, threshold_youden)
    kw        = threshold_kwargs or {}
    threshold = thr_fn(labels, probs, **kw)
    m_p       = _full_metrics_at(labels, probs, threshold)

    # Youden reference (always computed, backward compat)
    thr_youden = threshold_youden(labels, probs)
    m_y        = _full_metrics_at(labels, probs, thr_youden)

    # t05 backward compat (also covered by named strategy below)
    m_t05_bc = _full_metrics_at(labels, probs, 0.5)

    # ── 4 named strategies (Dewmika format) ──────────────────────────────────
    thr_rec  = threshold_min_recall(labels, probs, min_recall=rec_min_recall)
    thr_recp = threshold_min_precision(labels, probs, min_precision=recp_min_precision)
    thr_f1s  = threshold_max_f1(labels, probs)

    def _prefix(s: str, m: dict, thr: float) -> dict:
        return {
            f'{s}_thr':  thr,       f'{s}_tp':   m['tp'],
            f'{s}_fp':   m['fp'],   f'{s}_tn':   m['tn'],
            f'{s}_fn':   m['fn'],   f'{s}_acc':  m['acc'],
            f'{s}_prec': m['prec'], f'{s}_rec':  m['rec'],
            f'{s}_spec': m['spec'], f'{s}_npv':  m['npv'],
            f'{s}_f1':   m['f1'],   f'{s}_f2':   m['f2'],
        }

    result = {
        'tpr_5pct':           float(tpr_5pct),
        'auroc':              float(auroc),
        'auprc':              float(auprc),
        'threshold':          threshold,
        'threshold_strategy': threshold_strategy,
        'precision':          m_p['prec'],
        'recall':             m_p['rec'],
        'f1':                 m_p['f1'],
        'accuracy':           m_p['acc'],
        'tp':                 m_p['tp'],   'fp': m_p['fp'],
        'fn':                 m_p['fn'],   'tn': m_p['tn'],
        'threshold_youden':   thr_youden,
        'precision_youden':   m_y['prec'], 'recall_youden':   m_y['rec'],
        'f1_youden':          m_y['f1'],   'accuracy_youden': m_y['acc'],
        'tp_youden':          m_y['tp'],   'fp_youden':       m_y['fp'],
        'fn_youden':          m_y['fn'],   'tn_youden':       m_y['tn'],
        'precision_t05':      m_t05_bc['prec'], 'recall_t05':   m_t05_bc['rec'],
        'f1_t05':             m_t05_bc['f1'],   'accuracy_t05': m_t05_bc['acc'],
        'tp_t05':             m_t05_bc['tp'],   'fp_t05':       m_t05_bc['fp'],
        'fn_t05':             m_t05_bc['fn'],   'tn_t05':       m_t05_bc['tn'],
        'using_official':     OFFICIAL_METRICS_AVAILABLE,
        'num_permutations':   num_permutations,
        'n_pos':              n_pos,
        'n_total':            n_total,
    }
    result.update(_prefix('rec',  _full_metrics_at(labels, probs, thr_rec),  thr_rec))
    result.update(_prefix('recp', _full_metrics_at(labels, probs, thr_recp), thr_recp))
    result.update(_prefix('f1',   _full_metrics_at(labels, probs, thr_f1s),  thr_f1s))
    result.update(_prefix('t05',  m_t05_bc, 0.5))
    return result


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