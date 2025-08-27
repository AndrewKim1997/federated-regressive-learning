# frl/algo_frl.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple, Dict
import numpy as np

@dataclass
class ClientUpdate:
    delta: np.ndarray          # flat parameter delta
    num_samples: int
    dist: np.ndarray           # class distribution (K,)
    metadata: dict | None = None

def _safe_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum()

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m))))

def _wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    # Discrete 1D EMD via CDF L1; class order is 0..K-1
    p = _safe_prob(p)
    q = _safe_prob(q)
    cdf_diff = np.cumsum(p - q)
    return float(np.sum(np.abs(cdf_diff)))

def _pair_dist(p: np.ndarray, q: np.ndarray, metric: str) -> float:
    if metric == "js":
        return _js_divergence(p, q)
    elif metric == "wasserstein":
        return _wasserstein_1d(p, q)
    else:
        raise ValueError("metric must be 'wasserstein' or 'js'")

def compute_frl_weights(
    updates: List[ClientUpdate],
    *,
    ref: Literal["uniform", "empirical"] = "uniform",
    metric: Literal["wasserstein", "js"] = "wasserstein",
    size_power: float = 1.0,
    dist_power: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    K = len(updates[0].dist)
    if ref == "uniform":
        p_ref = np.full(K, 1.0 / K, dtype=float)
    elif ref == "empirical":
        p_ref = _safe_prob(np.mean([u.dist for u in updates], axis=0))
    else:
        raise ValueError("ref must be 'uniform' or 'empirical'")

    d = np.array([_pair_dist(u.dist, p_ref, metric) for u in updates], dtype=float)  # (M,)
    # Convert distances to affinity; smaller distance → larger weight
    a = np.exp(-dist_power * (d / (d.std() + 1e-12)))
    # Size prior
    n = np.array([u.num_samples for u in updates], dtype=float)
    n = (n / n.sum()) ** size_power
    w = a * (n / (n.sum() + 1e-12))
    beta = w / w.sum()

    info = {
        "ref_distribution": p_ref,
        "pairwise_distance": d,
        "weights": beta,
        "metric": metric,
        "ref": ref,
    }
    return beta.astype(float), info

def frl_aggregate(
    updates: List[ClientUpdate],
    *,
    ref: Literal["uniform", "empirical"] = "uniform",
    metric: Literal["wasserstein", "js"] = "wasserstein",
    size_power: float = 1.0,
    dist_power: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    beta, info = compute_frl_weights(updates, ref=ref, metric=metric,
                                     size_power=size_power, dist_power=dist_power)
    stacked = np.stack([u.delta.reshape(-1) for u in updates], axis=0)  # (M, D)
    agg = (beta[:, None] * stacked).sum(axis=0)
    return agg.reshape(updates[0].delta.shape), info
