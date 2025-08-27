from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

try:
    import torchvision  # type: ignore
    from torchvision import datasets, transforms as T  # type: ignore

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TORCHVISION = False


@dataclass
class NpDataset:
    """Simple NumPy dataset container."""
    X: np.ndarray  # [N, ...] float32
    y: np.ndarray  # [N] int64
    num_classes: int
    name: str
    split: str  # "train" | "test"


def _to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _scale_0_1(x: np.ndarray) -> np.ndarray:
    # For image-like inputs with 0..255 range
    x = _to_float32(x)
    if x.max() > 1.0:
        x = x / 255.0
    return x


# ---------------------------- MNIST ----------------------------

def load_mnist(
    root: Optional[str] = None,
    split: str = "train",
    as_numpy: bool = True,
) -> NpDataset:
    """
    MNIST loader with no hard dependency on torch/torchvision.
    - Primary path: sklearn.fetch_openml("mnist_784")
    - Split rule: first 60k → train, last 10k → test (OpenML order matches)
    - Returns flattened [N, 784] float32 by default; you can reshape to [N, 28, 28].

    Args:
        root: unused for OpenML path (kept for API symmetry).
        split: "train" or "test".
        as_numpy: kept for symmetry.

    Returns:
        NpDataset(X, y, num_classes=10, name="mnist", split=split)
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    # cache=False to avoid user cache issues in CI
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = data.data  # [70000, 784], uint8-like floats
    y = data.target  # strings like "0","1",...

    X = _scale_0_1(X.reshape(-1, 28, 28))  # [N,28,28], 0..1 float32
    y = y.astype(np.int64)

    if split == "train":
        X, y = X[:60000], y[:60000]
    else:
        X, y = X[60000:], y[60000:]

    # Flatten to [N,784] by default (model code can reshape back if needed)
    X = X.reshape(len(X), -1).astype(np.float32, copy=False)
    return NpDataset(X=X, y=y, num_classes=10, name="mnist", split=split)


# --------------------------- CIFAR-10 ---------------------------

def load_cifar10(
    root: Optional[str] = "./data",
    split: str = "train",
    as_numpy: bool = True,
) -> NpDataset:
    """
    CIFAR-10 loader.
    - Requires `torchvision`. If unavailable, raises a helpful error.
    - Output: [N, 32*32*3] float32 in [0,1], labels int64.

    Args:
        root: disk location for torchvision dataset cache.
        split: "train" or "test".
        as_numpy: returned arrays are numpy.

    Returns:
        NpDataset(X, y, num_classes=10, name="cifar10", split=split)
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    if not _HAS_TORCHVISION:
        raise RuntimeError(
            "CIFAR-10 requires torchvision. Install with:\n"
            "  pip install torchvision  # or torch+torchvision from PyTorch index"
        )

    is_train = (split == "train")
    tfm = T.Compose([T.ToTensor()])  # produces [C,H,W] in 0..1 float32
    ds = datasets.CIFAR10(root=root or "./data", train=is_train, transform=tfm, download=True)

    # Convert to numpy
    N = len(ds)
    X = np.empty((N, 32, 32, 3), dtype=np.float32)
    y = np.empty((N,), dtype=np.int64)
    for i in range(N):
        img, label = ds[i]  # img: torch.Tensor [C,H,W]
        X[i] = np.transpose(img.numpy(), (1, 2, 0))  # [H,W,C]
        y[i] = int(label)

    X = X.reshape(N, -1)  # [N, 3072]
    return NpDataset(X=X, y=y, num_classes=10, name="cifar10", split=split)


# --------------------------- Dispatcher -------------------------

def load_dataset(
    name: str,
    root: Optional[str] = None,
    split: str = "train",
    as_numpy: bool = True,
) -> NpDataset:
    """
    Unified dataset loader.
    Supported names: "mnist", "cifar10"
    """
    name_l = name.lower()
    if name_l == "mnist":
        return load_mnist(root=root, split=split, as_numpy=as_numpy)
    if name_l == "cifar10":
        return load_cifar10(root=root, split=split, as_numpy=as_numpy)
    raise ValueError(f"Unknown dataset '{name}'. Supported: mnist, cifar10")
