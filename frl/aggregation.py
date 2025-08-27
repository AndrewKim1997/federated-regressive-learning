from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ClientUpdate:
    """
    A single client's contribution for server aggregation.

    Attributes:
        delta: Parameter delta (e.g., model update) as a flat vector or ndarray of any shape.
        num_samples: Number of training samples used by the client.
        dist: Optional discrete distribution over K classes (probabilities, sum≈1) used by FRL.
        metadata: Arbitrary client metadata (id, scenario tags, etc.)
    """
    delta: np.ndarray
    num_samples: int
    dist: np.ndarray | None = None
    metadata: Dict = field(default_factory=dict)


def _stack_deltas(updates: Iterable[ClientUpdate]) -> Tuple[np.ndarray, Tuple[int, ...]]:
    shapes = [u.delta.shape for u in updates]
    if not shapes:
        raise ValueError("No client updates provided.")
    first = shapes[0]
    if any(s != first for s in shapes):
        raise ValueError(f"All client deltas must have the same shape. Got: {shapes}")
    flat = np.stack([u.delta.reshape(-1) for u in updates], axis=0)  # [M, D]
    return flat, first


def weighted_average(
    updates: List[ClientUpdate],
    weights: np.ndarray | None = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Generic weighted average aggregator.

    Args:
        updates: list of ClientUpdate.
        weights: shape [M] weights per client. If None, uses proportional to num_samples.
        normalize: if True, normalize weights to sum to 1.

    Returns:
        (global_delta, info)
        - global_delta: aggregated delta reshaped to original parameter shape.
        - info: {"weights": w, "weight_sum": float}
    """
    if not updates:
        raise ValueError("No updates to aggregate.")

    M = len(updates)
    if weights is None:
        w = np.array([max(0, u.num_samples) for u in updates], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (M,):
            raise ValueError(f"Weights must have shape ({M},), got {w.shape}.")

    if normalize:
        s = w.sum()
        if s <= 0:
            raise ValueError("Non-positive weight sum; cannot normalize.")
        w = w / s

    stacked, original_shape = _stack_deltas(updates)  # [M, D]
    global_flat = (w[:, None] * stacked).sum(axis=0)  # [D]
    return global_flat.reshape(original_shape), {"weights": w, "weight_sum": float(w.sum())}
