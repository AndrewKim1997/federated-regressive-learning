from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .aggregation import ClientUpdate, weighted_average


def fedavg_aggregate(updates: List[ClientUpdate]) -> Tuple[np.ndarray, Dict]:
    """
    FedAvg server aggregation: sample-size-weighted average of client deltas.

    Note:
        In classic FedAvg, weighting ∝ local data size.
    """
    return weighted_average(updates, weights=None, normalize=True)


def fedprox_aggregate(updates: List[ClientUpdate], mu: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    FedProx modifies the CLIENT objective with a proximal term.
    The SERVER aggregation remains a sample-size-weighted average like FedAvg.
    We expose mu for logging completeness only.

    Args:
        mu: proximal strength used by clients (logged only here).

    Returns:
        Same as fedavg_aggregate, with info["mu"] added.
    """
    delta, info = weighted_average(updates, weights=None, normalize=True)
    info["mu"] = mu
    return delta, info
