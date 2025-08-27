#!/usr/bin/env python3
# Inspect client class distributions and FRL weights for S1 on MNIST.

import yaml
import numpy as np
from frl.data.loaders import load_dataset
from frl.data.transforms import class_distribution
from frl.scenarios.scenario_gen import build_s1, build_s2, build_s3
from frl.algo_frl import compute_frl_weights
from frl.aggregation import ClientUpdate


def build_client_indices_from_config(y, cfg):
    sc = cfg.get("scenario", {})
    stype = sc.get("type", "").lower()
    seed = int(cfg.get("seed", 42))
    M = int(cfg.get("clients", {}).get("num_clients", 5))
    sizes = cfg.get("clients", {}).get("sizes", None)
    K = int(np.max(y)) + 1

    if stype in {"s1", "s1_equal_dist_diff_size"}:
        return build_s1(y, K, M, sizes, seed)
    elif stype in {"s2", "s2_hetero_dist_diff_size"}:
        alpha = sc.get("alpha", None)
        props = sc.get("class_props", None)
        return build_s2(y, K, M, sizes, seed, alpha=alpha, props_per_client=props)
    elif stype in {"s3", "s3_class_missing"}:
        base = sc.get("base", "iid")
        alpha = float(sc.get("alpha", 0.5))
        missing_map = sc.get("missing_map", None)
        fill_to_target = bool(sc.get("fill_to_target", False))
        return build_s3(y, K, M, sizes, seed, base=base, alpha=alpha, missing_map=missing_map, fill_to_target=fill_to_target)
    else:
        raise ValueError(f"Unknown scenario.type '{stype}'")


def main():
    ds = load_dataset("mnist", split="train")
    y, K = ds.y, ds.num_classes

    with open("frl/scenarios/s1_equal_dist_diff_size.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = build_client_indices_from_config(y, cfg)

    D = 8  # dummy delta length; only distributions matter for FRL weights
    updates = []
    for cid, idx in enumerate(splits):
        dist = class_distribution(y[idx], K)
        updates.append(ClientUpdate(delta=np.zeros(D, dtype=np.float32),
                                    num_samples=len(idx),
                                    dist=dist,
                                    metadata={"cid": cid}))

    beta, info = compute_frl_weights(
        updates,
        ref="uniform",         # or "empirical"
        metric="wasserstein",  # or "js"
        size_power=1.0,
        dist_power=1.0,
    )

    print("=== Client class distributions (first 3 decimals) ===")
    for u in updates:
        print(f"cid={u.metadata['cid']}, n={u.num_samples}, dist={np.round(u.dist,3)}")
    print("\n=== FRL weights (sum≈1) ===")
    for cid, b in enumerate(beta):
        print(f"cid={cid}: {b:.4f}")
    print("\nRef distribution:", np.round(info["ref_distribution"], 3))
    print("Pairwise distances:", np.round(info["pairwise_distance"], 4))


if __name__ == "__main__":
    main()
