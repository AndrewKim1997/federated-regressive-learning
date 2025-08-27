from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple accuracy."""
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Macro/micro/weighted precision, recall, f1.
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for multi-class.

    Args:
        probs: [N, K] predicted probabilities (softmaxed).
        labels: [N] integer labels in [0, K).
        n_bins: number of confidence bins.

    Returns:
        scalar ECE in [0, 1]
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError("probs must be [N, K]")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels length must match probs rows")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    # Bin by confidence
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    N = probs.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece_val += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece_val)


def _safe_normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = p.clip(eps, None)
    return p / p.sum()


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two discrete distributions.
    """
    p = _safe_normalize(p, eps)
    q = _safe_normalize(q, eps)
    m = 0.5 * (p + q)
    kl_pm = entropy(p, m, base=base)
    kl_qm = entropy(q, m, base=base)
    return float(0.5 * (kl_pm + kl_qm))


def wasserstein_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-D Wasserstein distance (Earth Mover's Distance) for discrete class distributions.

    Interprets classes on a line at positions 0..K-1 and uses class probabilities as weights.
    """
    p = _safe_normalize(p)
    q = _safe_normalize(q)
    K = len(p)
    positions = np.arange(K, dtype=float)
    return float(wasserstein_distance(positions, positions, u_weights=p, v_weights=q))
