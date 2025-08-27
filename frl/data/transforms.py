from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------ Basic helpers ------------------------

def to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def class_distribution(y: np.ndarray, num_classes: int, eps: float = 1e-12) -> np.ndarray:
    """
    Compute probability vector over classes.
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, eps, None)
    p = counts / counts.sum()
    return p.astype(np.float64, copy=False)


# --------------------- Federated partitioning -------------------

def iid_split(
    y: np.ndarray,
    num_clients: int,
    sizes: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    IID stratified split: each client gets the same class distribution as global,
    but (optionally) different sizes.

    Args:
        y: labels [N]
        num_clients: M
        sizes: optional list of sample counts per client. If None, as equal as possible.
        seed: rng seed

    Returns:
        list of index arrays per client
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    N = len(y)

    if sizes is None:
        base = N // num_clients
        sizes = [base] * num_clients
        for i in range(N - base * num_clients):
            sizes[i] += 1
    else:
        if sum(sizes) > N:
            raise ValueError("Sum(sizes) cannot exceed N")

    # stratify by class
    indices_by_class = [np.where(y == c)[0] for c in range(np.max(y) + 1)]
    for arr in indices_by_class:
        rng.shuffle(arr)

    # Proportional allocation per client
    p_global = class_distribution(y, num_classes=np.max(y) + 1)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    ptrs = [0] * len(indices_by_class)
    for m, target in enumerate(sizes):
        # target per class for this client
        per_c = np.floor(target * p_global).astype(int)
