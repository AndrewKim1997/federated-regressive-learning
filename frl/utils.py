from __future__ import annotations

import logging
import os
import random
from typing import Dict

import numpy as np


def set_seed(seed: int = 42, *, deterministic_hash: bool = True) -> None:
    """
    Set seed for Python, NumPy. (Extend here if using torch, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic_hash:
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "frl", level: int = logging.INFO) -> logging.Logger:
    """
    Simple console logger with consistent formatting.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", "%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def ensure_prob_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Clip and normalize to form a valid probability vector.
    """
    v = np.asarray(v, dtype=float)
    v = v.clip(eps, None)
    s = v.sum()
    if s <= 0:
        raise ValueError("Vector sum is non-positive; cannot normalize to probability.")
    return v / s
