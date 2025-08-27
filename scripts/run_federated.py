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
    clients_per_round: int | None = None,
    local_epochs: int = 1,
    batch_size: int = 128,
    lr: float = 0.1,
    seed: int = 42,
    aggregator: str = "frl",  # "fedavg" | "fedprox" | "frl"
    frl_ref: str = "uniform",  # "uniform" | "empirical"
    frl_metric: str = "wasserstein",  # "wasserstein" | "js"
    frl_size_power: float = 1.0,
    frl_dist_power: float = 1.0,
    log_dir: str = "results/logs",
) -> str:
    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    # Load data
    train = load_dataset(dataset_name, split="train")
    test = load_dataset(dataset_name, split="test")
    Xtr, ytr, K = train.X, train.y, train.num_classes
    Xte, yte = test.X, test.y

    # Build splits from scenario YAML/alias
    sc_cfg = load_scenario_alias_or_path(scenario)
    client_indices = build_client_indices_from_config(ytr, sc_cfg)
    M = len(client_indices)

    if clients_per_round is None or clients_per_round > M:
        clients_per_round = M

    # Init global model
    D = Xtr.shape[1]
    g_model = init_model(D, K, seed=seed)

    # Run
    rng = np.random.default_rng(seed)
    log_rows = []

    for rnd in range(1, rounds + 1):
        # sample participating clients
        part = rng.choice(M, size=clients_per_round, replace=False)
        updates: List[ClientUpdate] = []

        # compute client updates
        for cid in part:
            idx = client_indices[cid]
            Xc, yc = Xtr[idx], ytr[idx]
            # local copy
            local = SoftmaxModel(W=g_model.W.copy(), b=g_model.b.copy())
            for _ in range(local_epochs):
                sgd_epoch(Xc, yc, local, lr=lr, weight_decay=0.0, batch_size=batch_size, rng=rng)
            delta = model_params_vector(local) - model_params_vector(g_model)
            dist = class_distribution(yc, K)
            updates.append(ClientUpdate(delta=delta, num_samples=len(yc), dist=dist, metadata={"cid": int(cid)}))

        # aggregate
        if aggregator.lower() == "fedavg":
            agg_delta, agg_info = fedavg_aggregate(updates)
        elif aggregator.lower() == "fedprox":
            agg_delta, agg_info = fedprox_aggregate(updates, mu=0.0)
        elif aggregator.lower() == "frl":
            agg_delta, agg_info = frl_aggregate(
                updates,
                ref=frl_ref,
                metric=frl_metric,
                size_power=frl_size_power,
                dist_power=frl_dist_power,
            )
        else:
            raise ValueError("Unknown aggregator. Choose from: fedavg, fedprox, frl")

        apply_delta(g_model, agg_delta)

        # evaluate
        probs = forward_proba(Xte, g_model)
        yhat = probs.argmax(axis=1)
        acc = acc_fn(yte, yhat)
        ece = ece_fn(probs, yte, n_bins=15)

        row = {
            "round": rnd,
            "participants": int(len(part)),
            "aggregator": aggregator,
            "frl_ref": frl_ref if aggregator.lower() == "frl" else "",
            "frl_metric": frl_metric if aggregator.lower() == "frl" else "",
            "acc": acc,
            "ece": ece,
        }
        log_rows.append(row)
        print(f"[round {rnd:02d}] acc={acc:.4f} ece={ece:.4f}")

    # save log
    out_csv = os.path.join(log_dir, f"log_{dataset_name}_{os.path.basename(scenario).split('.')[0]}_{aggregator}.csv")
    pd.DataFrame(log_rows).to_csv(out_csv, index=False)
    return out_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Federated runner for FRL and baselines.")
    ap.add_argument("--dataset", type=str, default="mnist", help="mnist | cifar10")
    ap.add_argument("--scenario", type=str, default="frl/scenarios/s1_equal_dist_diff_size.yaml", help="path or alias (see configs/scenarios.yaml)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=None, help="clients per round (default: all clients in scenario)")
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggregator", type=str, default="frl", choices=["fedavg", "fedprox", "frl"])
    ap.add_argument("--frl-ref", type=str, default="uniform", choices=["uniform", "empirical"])
    ap.add_argument("--frl-metric", type=str, default="wasserstein", choices=["wasserstein", "js"])
    ap.add_argument("--frl-size-power", type=float, default=1.0)
    ap.add_argument("--frl-dist-power", type=float, default=1.0)
    ap.add_argument("--log-dir", type=str, default="results/logs")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_federated(
        dataset_name=args.dataset,
        scenario=args.scenario,
        rounds=args.rounds,
        clients_per_round=args.clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        aggregator=args.aggregator,
        frl_ref=args.frl_ref,
        frl_metric=args.frl_metric,
        frl_size_power=args.frl_size_power,
        frl_dist_power=args.frl_dist_power,
        log_dir=args.log_dir,
    )
