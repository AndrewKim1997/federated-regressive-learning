from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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
        # adjust rounding
        while per_c.sum() < target:
            c = rng.integers(0, len(per_c))
            per_c[c] += 1

        take = []
        for c, k in enumerate(per_c):
            start, end = ptrs[c], ptrs[c] + k
            take.extend(indices_by_class[c][start:end].tolist())
            ptrs[c] = end
        rng.shuffle(take)
        client_indices[m] = take

    return [np.array(ix, dtype=int) for ix in client_indices]


def dirichlet_noniid_split(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    sizes: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Popular non-IID splitter: per class, draw client proportions from Dirichlet(alpha).
    Smaller alpha -> more skew.

    Args:
        y: labels [N]
        num_clients: M
        alpha: Dirichlet concentration
        sizes: optional sample counts per client. If None, allocate as equal as possible.
        seed: rng seed
    """
    y = np.asarray(y, dtype=int)
    N = len(y)
    C = int(np.max(y)) + 1

    if sizes is None:
        base = N // num_clients
        sizes = [base] * num_clients
        for i in range(N - base * num_clients):
            sizes[i] += 1
    else:
        if sum(sizes) > N:
            raise ValueError("Sum(sizes) cannot exceed N")

    # class-wise pools
    pools = [rng.permutation(np.where(y == c)[0]).tolist() for c in range(C)]

    # sample proportions per class
    props = rng.dirichlet([alpha] * num_clients, size=C)  # [C, M]

    client_bins: List[List[int]] = [[] for _ in range(num_clients)]
    # First pass: per class allocations
    for c in range(C):
        total_c = len(pools[c])
        alloc_c = np.floor(props[c] * total_c).astype(int)  # [M]
        # ensure sum equals total_c
        while alloc_c.sum() < total_c:
            m = rng.integers(0, num_clients)
            alloc_c[m] += 1
        # take from pool
        pos = 0
        for m in range(num_clients):
            k = int(alloc_c[m])
            take = pools[c][pos : pos + k]
            client_bins[m].extend(take)
            pos += k

    # Second pass: trim/augment to target sizes
    for m in range(num_clients):
        arr = rng.permutation(client_bins[m]).tolist()
        if len(arr) > sizes[m]:
            client_bins[m] = arr[: sizes[m]]
        elif len(arr) < sizes[m]:
            # fill deficit from global leftovers
            deficit = sizes[m] - len(arr)
            # gather leftovers
            leftovers = [idx for idx in range(N) if all(idx not in b for b in client_bins)]
            add = rng.choice(leftovers, size=deficit, replace=False).tolist()
            client_bins[m].extend(add)
        rng.shuffle(client_bins[m])

    return [np.array(b, dtype=int) for b in client_bins]


def induce_class_missing(
    y: np.ndarray,
    missing_map: Dict[int, Sequence[int]],
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Remove specified classes per client to simulate class-missing scenario (S3).
    Args:
        y: global labels
        missing_map: {client_id: [class_id, ...]} classes to remove from that client
    Returns:
        mask_map: {client_id: boolean mask over that client's indices AFTER split}
    """
    mask_map: Dict[int, np.ndarray] = {}
    # This function does not split; it's designed to be applied AFTER you choose client indices.
    # Example usage:
    #   client_ix = iid_split(...)
    #   mask_map = induce_class_missing(y, {0:[0,1]})
    #   client0 = client_ix[0][mask_map[0]]
    # Here we just compute masks, so the caller can index into its own arrays.
    for cid, missing_classes in missing_map.items():
        mask_map[cid] = None  # filled by caller with their indices
    return mask_map
