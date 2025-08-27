from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


class UGEINotAvailable(RuntimeError):
    pass


def load_ugei_data(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for UGEI-like private medical dataset.
    This function NEVER accesses or ships real data.
    It only documents the expected interface and fails fast unless the user
    supplies their own, locally accessible dataset.

    Expected user-provided format (suggested, but you can adapt):
      - A single NPZ file with keys: "X" (float32, [N, D] or [N, H, W, C]),
        "y" (int64, [N]), and optional "classes" metadata.
      - OR a folder you handle in your own fork, then map into (X, y).

    Args:
        path: local filesystem path to a user-owned dataset file, e.g., "/home/me/ugei.npz".

    Returns:
        (X, y) as numpy arrays.

    Raises:
        UGEINotAvailable if path is None or file missing.
        ValueError if the loaded arrays are malformed.
    """
    if path is None:
        raise UGEINotAvailable(
            "UGEI dataset is not distributed with this repository due to privacy & contractual restrictions. "
            "Provide your OWN dataset path to proceed, e.g., load_ugei_data('/abs/path/to/ugei.npz')."
        )
    if not os.path.exists(path):
        raise UGEINotAvailable(f"UGEI dataset file not found at: {path}")

    if path.lower().endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        if "X" not in data or "y" not in data:
            raise ValueError("Missing keys in NPZ. Expected at least 'X' and 'y'.")
        X = data["X"]
        y = data["y"]
    else:
        raise UGEINotAvailable(
            "Unsupported UGEI path format. Provide a single NPZ file with arrays 'X' and 'y'."
        )

    # Basic sanity checks
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have mismatched first dimension: {X.shape[0]} vs {y.shape[0]}")
    if y.ndim != 1:
        raise ValueError("y must be 1-D integer labels.")
    if y.dtype.kind not in ("i", "u"):
        # coerce to int if it's safe
        y = y.astype(np.int64, copy=False)

    # Optional normalization: if pixel-like, scale to 0..1
    X = X.astype(np.float32, copy=False)
    if X.max() > 1.0:
        X = X / 255.0

    # Flatten images if needed
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return X, y
