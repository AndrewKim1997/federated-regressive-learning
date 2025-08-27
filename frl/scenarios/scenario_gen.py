from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
import pandas as pd

from frl.data.loaders import load_dataset
from frl.data.transforms import (
    class_distribution,
    dirichlet_noniid_split,
    iid_split,
)


@dataclass
class ScenarioConfig:
    name: str
    seed: int
    dataset: Dict  # {name: "mnist", split: "train", root: null}
    clients: Dict  # {num_clients: 5, sizes: [..](optional)}
    scenario: Dict  # {type: "s1_equal_dist_diff_size" | "s2_hetero_dist_diff_size" | "s3_class_missing", ...}


# ---------- helpers ----------

def _ensure_sizes(total_n: int, num_clients: int, sizes: Optional[Sequence[int]]) -> List[int]:
    if sizes is None:
        base = total_n // num_clients
        sizes = [base] * num_clients
        for i in range(total_n - base * num_clients):
            sizes[i] += 1
        return sizes
    sizes = list(map(int, sizes))
    if sum(sizes) > total_n:
        raise ValueError(f"Sum(sizes)={sum(sizes)} exceeds dataset size N={total_n}")
    return sizes


def _summarize_splits(y: np.ndarray, splits: List[np.ndarray], K: int) -> pd.DataFrame:
    rows = []
    for cid, idx in enumerate(splits):
        yy = y[idx]
        p = class_distribution(yy, K)
        rows.append({
            "client": cid,
            "n": int(len(idx)),
            "dist": p.tolist(),
        })
    df = pd.DataFrame(rows)
    return df


def _save_outputs(
    outdir: str,
    cfg: ScenarioConfig,
    client_indices: List[np.ndarray],
    dist_table: pd.DataFrame,
) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # indices.json
    indices_path = os.path.join(outdir, f"indices_{cfg.name}_{stamp}.json")
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump({str(i): idx.tolist() for i, idx in enumerate(client_indices)}, f)

    # summary.csv
    summary_path = os.path.join(outdir, f"summary_{cfg.name}_{stamp}.csv")
    dist_table.to_csv(summary_path, index=False)

    # echo config
    cfg_echo_path = os.path.join(outdir, f"config_echo_{cfg.name}_{stamp}.yaml")
    with open(cfg_echo_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "generated_at_utc": stamp,
            "config": {
                "name": cfg.name,
                "seed": cfg.seed,
                **cfg.dataset,
                **cfg.clients,
                **{"scenario": cfg.scenario},
            },
        }, f, sort_keys=False)

    return {
        "indices": indices_path,
        "summary": summary_path,
        "config_echo": cfg_echo_path,
    }


# ---------- custom splitter for fixed per-client class proportions ----------

def split_by_target_props(
    y: np.ndarray,
    sizes: Sequence[int],
    props_per_client: Sequence[Sequence[float]],
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Split dataset into clients with explicit target size per client AND
    explicit class proportion vector per client.

    Args:
        y: [N] global labels
        sizes: [M] target samples per client
        props_per_client: [M, K] rows sum to 1 (we will renormalize for safety)

    Returns:
        list of index arrays per client
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    N = len(y)
    M = len(sizes)
    K = int(np.max(y)) + 1

    # pools per class
    pools = [rng.permutation(np.where(y == c)[0]).tolist() for c in range(K)]

    # sanitize props
    P = np.asarray(props_per_client, dtype=float)
    if P.shape != (M, K):
        raise ValueError(f"props_per_client must be [M,K]={M,K}, got {P.shape}")
    P = np.clip(P, 1e-12, None)
    P = P / P.sum(axis=1, keepdims=True)

    # allocate per client
    client_bins: List[List[int]] = [[] for _ in range(M)]
    for m in range(M):
        target = int(sizes[m])
        # integer target per class with rounding fix
        per_c = np.floor(P[m] * target).astype(int)
        while per_c.sum() < target:
            c = int(rng.integers(0, K))
            per_c[c] += 1

        # take indices from class pools
        take = []
        for c in range(K):
            k = int(per_c[c])
            if k == 0:
                continue
            if k > len(pools[c]):
                # not enough in pool; fill what we can, remainder later
                k = len(pools[c])
            take.extend(pools[c][:k])
            pools[c] = pools[c][k:]
        rng.shuffle(take)
        client_bins[m] = take

    # if there is any deficit due to pool exhaustion, top up randomly from leftovers
    assigned = set(idx for bin_ in client_bins for idx in bin_)
    leftovers = [idx for idx in range(N) if idx not in assigned]
    rng.shuffle(leftovers)
    ptr = 0
    for m in range(M):
        deficit = sizes[m] - len(client_bins[m])
        if deficit > 0:
            add = leftovers[ptr:ptr + deficit]
            ptr += deficit
            client_bins[m].extend(add)
        rng.shuffle(client_bins[m])

    return [np.array(b, dtype=int) for b in client_bins]


# ---------- scenario builders ----------

def build_s1(y: np.ndarray, K: int, num_clients: int, sizes: Optional[Sequence[int]], seed: int) -> List[np.ndarray]:
    sizes_ = _ensure_sizes(len(y), num_clients, sizes)
    return iid_split(y=y, num_clients=num_clients, sizes=sizes_, seed=seed)


def build_s2(
    y: np.ndarray,
    K: int,
    num_clients: int,
    sizes: Optional[Sequence[int]],
    seed: int,
    *,
    alpha: Optional[float] = None,
    props_per_client: Optional[Sequence[Sequence[float]]] = None,
) -> List[np.ndarray]:
    sizes_ = _ensure_sizes(len(y), num_clients, sizes)
    if props_per_client is not None:
        return split_by_target_props(y, sizes_, props_per_client, seed=seed)
    # default: Dirichlet non-iid
    alpha = 0.5 if alpha is None else float(alpha)
    return dirichlet_noniid_split(y=y, num_clients=num_clients, alpha=alpha, sizes=sizes_, seed=seed)


def build_s3(
    y: np.ndarray,
    K: int,
    num_clients: int,
    sizes: Optional[Sequence[int]],
    seed: int,
    *,
    base: str = "iid",
    alpha: float = 0.5,
    missing_map: Optional[Dict[int, Sequence[int]]] = None,
    fill_to_target: bool = False,
) -> List[np.ndarray]:
    sizes_ = _ensure_sizes(len(y), num_clients, sizes)
    if base == "iid":
        splits = iid_split(y=y, num_clients=num_clients, sizes=sizes_, seed=seed)
    elif base == "dirichlet":
        splits = dirichlet_noniid_split(y=y, num_clients=num_clients, alpha=alpha, sizes=sizes_, seed=seed)
    else:
        raise ValueError("base must be 'iid' or 'dirichlet'")

    if not missing_map:
        return splits

    rng = np.random.default_rng(seed)
    new_splits: List[np.ndarray] = []
    global_pool = [idx for idx in range(len(y)) if all(idx not in s for s in splits)]  # usually empty
    rng.shuffle(global_pool)
    pool_ptr = 0

    for cid, idx in enumerate(splits):
        yy = y[idx]
        keep_mask = np.ones(len(idx), dtype=bool)
        miss = set(int(c) for c in missing_map.get(cid, []))
        if miss:
            for j, lab in enumerate(yy):
                if int(lab) in miss:
                    keep_mask[j] = False
        kept = idx[keep_mask]

        if fill_to_target:
            deficit = sizes_[cid] - len(kept)
            if deficit > 0:
                add = global_pool[pool_ptr: pool_ptr + deficit]
                pool_ptr += deficit
                kept = np.concatenate([kept, np.array(add, dtype=int)], axis=0)

        new_splits.append(kept)

    return new_splits


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Programmatic federated scenario generator (S1/S2/S3).")
    ap.add_argument("-c", "--config", required=True, help="Path to scenario YAML (see s1/s2/s3 examples).")
    ap.add_argument("-o", "--outdir", default="results/scenarios", help="Output directory for JSON/CSV/YAML echo.")
    ap.add_argument("--preview", action="store_true", help="Print summary table to stdout.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ScenarioConfig(
        name=str(raw.get("name", "scenario")),
        seed=int(raw.get("seed", 42)),
        dataset=dict(raw.get("dataset", {})),
        clients=dict(raw.get("clients", {})),
        scenario=dict(raw.get("scenario", {})),
    )

    ds_name = cfg.dataset.get("name", "mnist")
    ds_split = cfg.dataset.get("split", "train")
    ds_root = cfg.dataset.get("root", None)

    data = load_dataset(ds_name, root=ds_root, split=ds_split)
    X, y, K = data.X, data.y, int(data.num_classes)
    M = int(cfg.clients.get("num_clients", 5))
    sizes = cfg.clients.get("sizes", None)

    stype = cfg.scenario.get("type", "").lower()
    if stype in {"s1", "s1_equal_dist_diff_size"}:
        splits = build_s1(y, K, M, sizes, seed=cfg.seed)
    elif stype in {"s2", "s2_hetero_dist_diff_size"}:
        alpha = cfg.scenario.get("alpha", None)
        props = cfg.scenario.get("class_props", None)
        splits = build_s2(y, K, M, sizes, seed=cfg.seed, alpha=alpha, props_per_client=props)
    elif stype in {"s3", "s3_class_missing"}:
        base = cfg.scenario.get("base", "iid")
        alpha = float(cfg.scenario.get("alpha", 0.5))
        missing_map = cfg.scenario.get("missing_map", None)
        fill_to_target = bool(cfg.scenario.get("fill_to_target", False))
        splits = build_s3(y, K, M, sizes, seed=cfg.seed, base=base, alpha=alpha,
                          missing_map=missing_map, fill_to_target=fill_to_target)
    else:
        raise ValueError(f"Unknown scenario.type='{stype}'")

    df = _summarize_splits(y, splits, K)
    paths = _save_outputs(args.outdir, cfg, splits, df)

    if args.preview:
        print(df.to_string(index=False))
        print("\nSaved:")
        for k, v in paths.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
