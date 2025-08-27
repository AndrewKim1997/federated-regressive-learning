#!/usr/bin/env python3
# Inspect client class distributions and FRL weights for S1 on MNIST.

import yaml
import numpy as np
from frl.data.loaders import load_dataset
from frl.data.transforms import class_distribution
from frl.scenarios.scenario_gen import build_client_indices_from_config
from frl.algo_frl import compute_frl_weights
from frl.aggregation import ClientUpdate

def main():
    # Load dataset (train split)
    ds = load_dataset("mnist", split="train")
    X, y, K = ds.X, ds.y, ds.num_classes

    # Load scenario config (S1)
    with open("frl/scenarios/s1_equal_dist_diff_size.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = build_client_indices_from_config(y, cfg)

    # Build dummy deltas and per-client class distributions
    # (Only distributions matter for FRL weights; deltas are placeholders)
    D = 8  # arbitrary parameter length for dummy deltas
    updates = []
    for cid, idx in enumerate(splits):
        dist = class_distribution(y[idx], K)
        updates.append(
            ClientUpdate(
                delta=np.zeros(D, dtype=np.float32),
                num_samples=len(idx),
                dist=dist,
                metadata={"cid": cid},
            )
        )

    # Compute FRL weights
    beta, info = compute_frl_weights(
        updates,
        ref="uniform",         # or "empirical"
        metric="wasserstein",  # or "js"
        size_power=1.0,
        dist_power=1.0,
    )

    # Pretty print
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
