import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml
import pytest

# Import the runner module so we can monkeypatch its load_dataset symbol
import scripts.run_federated as runner


@dataclass
class FakeDataset:
    X: np.ndarray
    y: np.ndarray
    num_classes: int
    name: str = "fake"
    split: str = "train"


def _make_fake_dataset(n, d, k, seed=0):
    """Linearly separable toy dataset (NumPy only)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=2.0, size=(k, d))
    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        c = rng.integers(0, k)
        X[i] = rng.normal(loc=means[c], scale=1.0)
        y[i] = c
    return X, y


def fake_load_dataset(name: str, split: str = "train", **kwargs):
    """Replacement for frl.data.loaders.load_dataset used inside the runner."""
    if split == "train":
        X, y = _make_fake_dataset(n=300, d=20, k=3, seed=1)
    else:
        X, y = _make_fake_dataset(n=120, d=20, k=3, seed=2)
    return FakeDataset(X=X, y=y, num_classes=3, name=f"fake-{name}", split=split)


def _write_minimal_s1_yaml(path, num_clients=3, sizes=(100, 100, 100), seed=42):
    cfg = {
        "name": "s1_minimal",
        "seed": seed,
        "dataset": {"name": "fake", "split": "train"},
        "clients": {"num_clients": num_clients, "sizes": list(sizes)},
        "scenario": {"type": "s1_equal_dist_diff_size"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def test_minirun_with_fake_data(monkeypatch, tmp_path):
    # Monkeypatch loader inside the runner to avoid any network or torchvision dependency
    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)

    # Create a temporary scenario YAML whose sizes match our fake dataset
    sc_path = tmp_path / "s1_minimal.yaml"
    _write_minimal_s1_yaml(sc_path)

    # Run a tiny training (2 rounds) and write logs under tmp_path
    out_csv = runner.train_federated(
        dataset_name="mnist",  # ignored by our fake loader
        scenario=str(sc_path),
        rounds=2,
        clients_per_round=3,
        local_epochs=1,
        batch_size=64,
        lr=0.1,
        seed=123,
        aggregator="frl",
        log_dir=str(tmp_path),
    )

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    # Basic sanity checks on the log schema and values
    assert set(["round", "acc", "ece", "aggregator"]).issubset(df.columns)
    assert df["round"].iloc[-1] == 2
    assert (df["acc"] >= 0).all() and (df["acc"] <= 1).all()
