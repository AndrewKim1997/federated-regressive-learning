# frl/metrics.py
from __future__ import annotations
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.any():
            ece_val += np.abs(acc[m].mean() - conf[m].mean()) * (m.sum() / len(y_true))
    return float(ece_val)

def _safe_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum()

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps); q = _safe_prob(q, eps)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m)))))

def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "macro",   # "macro" | "micro" | "none"
    num_classes: int | None = None,
    eps: float = 1e-12,
):
    """
    Return (precision, recall, f1).
    - macro: average over classes
    - micro: global TP/FP/FN
    - none : per-class arrays
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    # per-class counts
    tp = np.zeros(num_classes, dtype=float)
    fp = np.zeros(num_classes, dtype=float)
    fn = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        tp[c] = float(np.sum((y_true == c) & (y_pred == c)))
        fp[c] = float(np.sum((y_true != c) & (y_pred == c)))
        fn[c] = float(np.sum((y_true == c) & (y_pred != c)))

    if average == "micro":
        TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
        prec = TP / (TP + FP + eps)
        rec  = TP / (TP + FN + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        return float(prec), float(rec), float(f1)

    # per-class
    prec_c = tp / (tp + fp + eps)
    rec_c  = tp / (tp + fn + eps)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    if average == "none":
        return prec_c, rec_c, f1_c
    # macro
    return float(np.mean(prec_c)), float(np.mean(rec_c)), float(np.mean(f1_c))
