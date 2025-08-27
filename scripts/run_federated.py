#!/usr/bin/env python3
"""
Minimal federated runner (NumPy-based) for FRL and baselines.
- Dataset: MNIST (via scikit-learn OpenML) by default; CIFAR-10 requires torchvision.
- Model: multiclass logistic regression (softmax) trained with SGD on each client.
- Aggregators: FedAvg, FedProx (server-side same as FedAvg), FRL (Wasserstein/JS-based).
- Scenario: load S1/S2/S3 YAML or alias path (see configs/scenarios.yaml).
- Outputs: CSV log under results/logs and prints a short progress line per round.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from frl import (
    ClientUpdate,
    fedavg_aggregate,
    fedprox_aggregate,
    frl_aggregate,
    accuracy as acc_fn,
    ece as ece_fn,
    set_seed,
)
from frl.data.loaders import load_dataset
from frl.data.transforms import class_distribution
from frl.scenarios.scenario_gen import (
    build_s1,
    build_s2,
    build_s3,
)


# ---------------------- simple softmax model ----------------------

@dataclass
class SoftmaxModel:
    W: np.ndarray  # [D, K]
    b: np.ndarray  # [K]


def init_model(input_dim: int, num_classes: int, seed: int = 42) -> SoftmaxModel:
    rng = np.random.default_rng(seed)
    # Small random init
    W = rng.normal(scale=0.01, size=(input_dim, num_classes)).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)
    return SoftmaxModel(W=W, b=b)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def forward_proba(X: np.ndarray, model: SoftmaxModel) -> np.ndarray:
    logits = X @ model.W + model.b[None, :]
    return softmax(logits)


def onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def sgd_epoch(
    X: np.ndarray,
    y: np.ndarray,
    model: SoftmaxModel,
    lr: float = 1e-1,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(0)
    N, D = X.shape
    K = model.b.shape[0]
    idx = rng.permutation(N)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        batch = idx[start:end]
        Xb, yb = X[batch], y[batch]

        P = forward_proba(Xb, model)                # [B, K]
        Y = onehot(yb, K)                            # [B, K]
        # cross-entropy gradient
        G = (P - Y) / float(len(yb))                 # [B, K]
        grad_W = Xb.T @ G + weight_decay * model.W   # [D, K]
        grad_b = G.sum(axis=0)                       # [K]

        model.W -= lr * grad_W
        model.b -= lr * grad_b


def model_params_vector(model: SoftmaxModel) -> np.ndarray:
    return np.concatenate([model.W.reshape(-1), model.b], axis=0)


def apply_delta(model: SoftmaxModel, delta: np.ndarray) -> None:
    D, K = model.W.shape
    dW = delta[: D * K].reshape(D, K)
    db = delta[D * K :]
    model.W += dW
    model.b += db


# ---------------------- scenario loading ----------------------

def load_scenario_alias_or_path(alias_or_path: str) -> dict:
    """
    Accepts either a path to a scenario YAML (frl/scenarios/*.yaml) or an alias
    defined in configs/scenarios.yaml (maps to a path).
    """
    if alias_or_path.endswith(".yaml") or alias_or_path.endswith(".yml"):
        with open(alias_or_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # try alias
    with open("configs/scenarios.yaml", "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    if alias_or_path in mapping:
        path = mapping[alias_or_path]
    else:
        path = mapping.get("presets", {}).get(alias_or_path, {}).get("alias", None)
        if path is not None:  # preset alias resolves to another alias -> path
            with open("configs/scenarios.yaml", "r", encoding="utf-8") as f:
                mapping2 = yaml.safe_load(f)
            path = mapping2[path]
    if not path:
        raise ValueError(f"Unknown scenario alias '{alias_or_path}'.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client_indices_from_config(y: np.ndarray, cfg: dict) -> List[np.ndarray]:
    sc = cfg.get("scenario", {})
    stype = sc.get("type", "").lower()
    seed = int(cfg.get("seed", 42))
    M = int(cfg.get("clients", {}).get("num_clients", 5))
    sizes = cfg.get("clients", {}).get("sizes", None)
    K = int(np.max(y)) + 1

    if stype in {"s1", "s1_equal_dist_diff_size"}:
        splits = build_s1(y, K, M, sizes, seed)
    elif stype in {"s2", "s2_hetero_dist_diff_size"}:
        alpha = sc.get("alpha", None)
        props = sc.get("class_props", None)
        splits = build_s2(y, K, M, sizes, seed, alpha=alpha, props_per_client=props)
    elif stype in {"s3", "s3_class_missing"}:
        base = sc.get("base", "iid")
        alpha = float(sc.get("alpha", 0.5))
        missing_map = sc.get("missing_map", None)
        fill_to_target = bool(sc.get("fill_to_target", False))
        splits = build_s3(y, K, M, sizes, seed, base=base, alpha=alpha, missing_map=missing_map, fill_to_target=fill_to_target)
    else:
        raise ValueError(f"Unknown scenario.type '{stype}'")
    return splits


# ---------------------- federated loop ----------------------

def train_federated(
    dataset_name: str = "mnist",
    scenario: str = "frl/scenarios/s1_equal_dist_diff_size.yaml",
    rounds: int = 3,
    clie
