# frl/data/transforms.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def class_distribution(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Return normalized class histogram p \in R^K for labels y.
    """
    y = np.asarray(y, dtype=int)
    K = int(num_classes)
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    total = float(cnt.sum())
    if total <= 0:
        return np.full(K, 1.0 / K, dtype=np.float64)
    return (cnt / total).astype(np.float64)


def _ensure_sizes(total_n: int, num_clients: int, sizes: Optional[Sequence[int]]) -> np.ndarray:
    """
    Make a size vector of length M that sums to <= total_n. If sizes is None,
    split as evenly as possible with +1 remainders for the first few clients.
    """
    if sizes is None:
        base = total_n // num_clients
        rem = total_n - base * num_clients
        out = np.full(num_clients, base, dtype=int)
        out[:rem] += 1
        return out
    arr = np.asarray(sizes, dtype=int)
    if arr.ndim != 1 or len(arr) != num_clients:
        raise ValueError("sizes must be a 1-D sequence with length num_clients")
    if int(arr.sum()) > total_n:
        raise ValueError(f"sum(sizes)={int(arr.sum())} exceeds dataset size N={total_n}")
    return arr


def iid_split(
    y: np.ndarray,
    num_clients: int,
    sizes: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Stratified IID-like split: each client roughly follows the global distribution.
    - y: global labels [N]
    - sizes: per-client target sizes (len=M). If None, split almost evenly.
    - Returns: list of index arrays per client
    """
    y = np.asarray(y, dtype=int)
    N = int(len(y))
    K = int(np.max(y)) + 1
    M = int(num_clients)
    rng = np.random.default_rng(seed)

    target = _ensure_sizes(N, M, sizes)  # (M,)
    p_global = class_distribution(y, K)  # (K,)

    # Build per-class index pools (shuffled)
    indices_by_class = [rng.permutation(np.where(y == c)[0]).tolist() for c in range(K)]
    ptrs = [0] * K
    client_bins: List[List[int]] = [[] for _ in range(M)]

    for m in range(M):
        n_m = int(target[m])
        # initial integer allocation by rounding
        alloc = np.floor(p_global * n_m).astype(int)
        # fix rounding to hit exact n_m
        while int(alloc.sum()) < n_m:
            c = int(rng.integers(0, K))
            alloc[c] += 1

        take: List[int] = []
        for c in range(K):
            k_c = int(alloc[c])
            if k_c <= 0:
                continue
            start = ptrs[c]
            end = min(start + k_c, len(indices_by_class[c]))
            if end > start:
                take.extend(indices_by_class[c][start:end])
                ptrs[c] = end

        # If some class pools were short, top-up from any remaining pool
        deficit = n_m - len(take)
        if deficit > 0:
            leftovers = []
            for c in range(K):
                if ptrs[c] < len(indices_by_class[c]):
                    leftovers.extend(indices_by_class[c][ptrs[c] :])
                    ptrs[c] = len(indices_by_class[c])
            if leftovers:
                rng.shuffle(leftovers)
                take.extend(leftovers[:deficit])

        rng.shuffle(take)
        client_bins[m] = take[:n_m]

    return [np.array(b, dtype=int) for b in client_bins]


def dirichlet_noniid_split(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    sizes: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Heterogeneous split via Dirichlet class proportions per class.
    - For each class c, draw proportions over clients ~ Dir(alpha).
    - Allocate integer counts per (c, client) and pull from per-class pools.
    - If sizes is provided, perform a second pass to trim/augment to targets.

    Returns list of index arrays per client.
    """
    y = np.asarray(y, dtype=int)
    N = int(len(y))
    K = int(np.max(y)) + 1
    M = int(num_clients)
    rng = np.random.default_rng(seed)

    target = _ensure_sizes(N, M, sizes)  # may be near-even if None

    # Per-class pools
    pools = [rng.permutation(np.where(y == c)[0]).tolist() for c in range(K)]

    # Sample class->client proportions
    props = rng.dirichlet([alpha] * M, size=K)  # shape [K, M]

    client_bins: List[List[int]] = [[] for _ in range(M)]

    # First pass: assign by class
    for c in range(K):
        total_c = len(pools[c])
        if total_c == 0:
            continue
        alloc = np.floor(props[c] * total_c).astype(int)  # (M,)
        while int(alloc.sum()) < total_c:
            m = int(rng.integers(0, M))
            alloc[m] += 1

        start = 0
        for m in range(M):
            k = int(alloc[m])
            if k <= 0:
                continue
            end = start + k
            take = pools[c][start:end]
            client_bins[m].extend(take)
            start = end

    # Second pass: adjust to target sizes if provided
    if sizes is not None:
        # trim or top-up to exactly target[m]
        all_assigned = set(idx for bin_ in client_bins for idx in bin_)
        leftovers = [i for i in range(N) if i not in all_assigned]
        rng.shuffle(leftovers)
        lp = 0

        for m in range(M):
            arr = rng.permutation(client_bins[m]).tolist()
            n_m = int(target[m])
            if len(arr) > n_m:
                client_bins[m] = arr[:n_m]
            elif len(arr) < n_m:
                deficit = n_m - len(arr)
                add = leftovers[lp : lp + deficit]
                lp += deficit
                client_bins[m] = arr + add
            else:
                client_bins[m] = arr

    # Final shuffle per client for randomness
    for m in range(M):
        rng.shuffle(client_bins[m])

    return [np.array(b, dtype=int) for b in client_bins]
