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
    with open(indices_path,_
