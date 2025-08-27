from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Literal, Tuple

import numpy as np

from .aggregation import ClientUpdate, weighted_average
from .metrics import js_divergence, wasserstein_discrete
from .utils import ensure_prob_vector


DistanceName = Literal["wasserstein", "js"]


def _infer_ref_distribution(
    updates: List[ClientUpdate],
    ref: Literal["uniform", "empirical"] | np.ndarray = "uniform",
) -> np.ndarray:
    """
    Build reference distribution f:
      - "uniform": uniform over K classes
      - "empirical": sample-size-weighted mean of clients' distributions
      - ndarray: use directly (will be normalized)
    """
    # collect
    dists = [u.dist for u in updates if u.dist is not None]
    if not dists:
        raise ValueError("FRL requires client 'dist' arrays; none were provided.")
    K = len(dists[0])
    if any(len(d) != K for d in dists):
        lens = [len(d) for d in dists]
        raise ValueError(f"Inconsistent dist lengths; expected {K}, got {lens}")

    if isinstance(ref, np.ndarray):
        f = ensure_prob_vector(ref)
        if len(f) != K:
            raise ValueError(f"Provided ref distribution length {len(f)} != K={K}")
        return f

    if ref == "uniform":
        return np.ones(K, dtype=float) / K

    if ref == "empirical":
        sizes = np.array([max(0, u.num_samples) for u in updates], dtype=float)
        sizes = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(len(updates)) / len(updates)
        mat = np.stack([ensure_prob_vector(u.dist) for u in updates], axis=0)  # [M, K]
        return (sizes[:, None] * mat).sum(axis=0)

    raise ValueError(f"Unknown ref='{ref}'")


def _pairwise_distance(
    f: np.ndarray,
    g: np.ndarray,
    metric: DistanceName = "wasserstein",
) -> float:
    if metric == "wasserstein":
        return wasserstein_discrete(f, g)
    if metric == "js":
        return js_divergence(f, g)
    raise ValueError(f"Unknown metric={metric}")


def compute_frl_weights(
    updates: List[ClientUpdate],
    ref: Literal["uniform", "empirical"] | np.ndarray = "uniform",
    metric: DistanceName = "wasserstein",
    *,
    size_power: float = 1.0,
    dist_power: float = 1.0,
    eps: float = 1e-8,
    clip_min: float = 1e-8,
    clip_max: float | None = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute FRL weights β_i ∝ (num_samples_i^size_power) * (1 / distance(f, g_i)^dist_power)

    Args:
        updates: list of ClientUpdate (must have dist and num_samples).
        ref: reference distribution f – "uniform", "empirical", or explicit ndarray.
        metric: distance function between f and g_i ("wasserstein" or "js").
        size_power: exponent for sample-size scaling (1.0 → proportional to size).
        dist_power: exponent for distance penalty (1.0 → inverse distance).
        eps: small constant added inside the inverse to avoid division-by-zero.
        clip_min: floor for β_i before normalization.
        clip_max: optional cap for β_i before normalization.
        normalize: if True, rescale β to sum to 1.

    Returns:
        weights β (shape [M]), and info dict with diagnostics.
    """
    if not updates:
        raise ValueError("No client updates provided.")
    f = _infer_ref_distribution(updates, ref=ref)

    M = len(updates)
    sizes = np.array([max(0, u.num_samples) for u in updates], dtype=float)
    sizes = np.power(sizes, size_power)

    dists = []
    for u in updates:
        if u.dist is None:
            raise ValueError("Each update must include 'dist' for FRL.")
        d = _pairwise_distance(f, ensure_prob_vector(u.dist), metric=metric)
        dists.append(d)
    dists = np.asarray(dists, dtype=float)

    inv_term = 1.0 / np.power(dists + eps, dist_power)
    beta = sizes * inv_term

    if clip_max is not None:
        beta = np.minimum(beta, clip_max)
    beta = np.maximum(beta, clip_min)

    if normalize:
        s = beta.sum()
        if s <= 0:
            raise ValueError("Non-positive β sum after clipping.")
        beta = beta / s

    info = {
        "ref_distribution": f,
        "pairwise_distance": dists,
        "size_term": sizes,
        "beta_raw": beta.copy() if normalize else beta.copy(),
        "metric": metric,
        "size_power": size_power,
        "dist_power": dist_power,
    }
    return beta, info


def frl_aggregate(
    updates: List[ClientUpdate],
    *,
    ref: Literal["uniform", "empirical"] | np.ndarray = "uniform",
    metric: DistanceName = "wasserstein",
    size_power: float = 1.0,
    dist_power: float = 1.0,
    eps: float = 1e-8,
    clip_min: float = 1e-8,
    clip_max: float | None = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute FRL weights and aggregate client deltas accordingly.

    Returns:
        (global_delta, info)
        - global_delta: aggregated parameter delta
        - info: dict with 'weights', 'pairwise_distance', and other diagnostics
    """
    beta, info = compute_frl_weights(
        updates,
        ref=ref,
        metric=metric,
        size_power=size_power,
        dist_power=dist_power,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
        normalize=True,  # always normalize before aggregation
    )
    global_delta, agg_info = weighted_average(updates, weights=beta, normalize=True)
    info.update(agg_info)
    info["weights"] = beta
    return global_delta, info
