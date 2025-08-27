from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

# Reuse the same dataclass from algo_frl to avoid type duplication
try:
    from .algo_frl import ClientUpdate  # delta: np.ndarray, num_samples: int, dist: np.ndarray, metadata: dict|None
except Exception:
    # Minimal fallback to avoid circular import during static analysis
    from dataclasses import dataclass
    @dataclass
    class ClientUpdate:
        delta: np.ndarray
        num_samples: int
        dist: np.ndarray
        metadata: dict | None = None


def _weights_by_size(updates: List[ClientUpdate]) -> np.ndarray:
    n = np.array([u.num_samples for u in updates], dtype=float)
    w = n / (n.sum() + 1e-12)
    return w


def fedavg_aggregate(updates: List[ClientUpdate]) -> Tuple[np.ndarray, Dict]:
    """
    Standard FedAvg server aggregation: weighted average of client deltas by sample count.
    Returns (aggregated_delta, info_dict).
    """
    assert len(updates) > 0, "No client updates"
    w = _weights_by_size(updates)               # (M,)
    stacked = np.stack([u.delta.reshape(-1) for u in updates], axis=0)  # (M, D)
    agg = (w[:, None] * stacked).sum(axis=0)    # (D,)
    info = {"weights": w}
    return agg.reshape(updates[0].delta.shape), info


def fedprox_aggregate(updates: List[ClientUpdate], mu: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    FedProx uses the same server-side averaging as FedAvg; 'mu' affects the CLIENT objective only.
    We expose 'mu' to make the signature explicit, but aggregation is size-weighted average.
    """
    return fedavg_aggregate(updates)
