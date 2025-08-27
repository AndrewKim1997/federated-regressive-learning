# Reproducibility Guide

This repository focuses on **method reproducibility** for Federated Regressive Learning (FRL). The exact experimental conditions from the 2024 paper are not fully recoverable; instead, we provide **seeded, programmatic scenarios** and **deterministic pipelines** to reproduce relative trends.

---

## 1) Environment

### Option A: Conda

```bash
conda env create -f env/environment.yml
conda activate frl
# (optional) Install PyTorch CPU wheels if you plan to use torchvision-based loaders:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
```

### Option B: pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Option C: Docker (CPU)

```bash
docker build -f docker/Dockerfile -t frl:cpu .
docker run --rm -it -v "$PWD:/app" frl:cpu python -c "import frl; print('ok')"
```

---

## 2) Determinism & Seeds

Randomness is controlled via Python/NumPy seeds. The provided runner uses a NumPy-only softmax model (no PyTorch dependency for training).

* `frl.utils.set_seed(seed=42)` sets Python/NumPy and `PYTHONHASHSEED`.
* Scenario YAMLs include a `seed` field; change it to regenerate splits.

If you extend models to PyTorch:

```python
import os, random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.use_deterministic_algorithms(True)
```

---

## 3) Datasets

* **MNIST**: loaded via scikit-learn (OpenML). Arrays are `float32` in `[0,1]`.
* **CIFAR-10**: requires `torchvision` for download; converted to NumPy for training.
* **UGEI**: **not distributed**. See `docs/PRIVACY.md` and `frl/data/ugei_placeholder.py`.

> Never commit datasets. `.gitignore` excludes `data/` and large artifacts.

---

## 4) Scenarios (S1/S2/S3)

Generate splits from YAML:

```bash
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s1_equal_dist_diff_size.yaml \
  -o results/scenarios --preview
```

Artifacts:

* `indices_*.json` — client index lists,
* `summary_*.csv` — per-client class distributions,
* `config_echo_*.yaml` — exact config echo with timestamp.

See `docs/SCENARIOS.md` for definitions.

---

## 5) Minimal End-to-End Run

```bash
python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures
```

---

## 6) Version Pinning & Provenance

```bash
python -m pip freeze > results/requirements-$(date -u +%Y%m%dT%H%M%SZ).txt
python --version   > results/python-version.txt
git rev-parse HEAD > results/git-commit.txt
# Optional:
git diff --no-color > results/git-diff.patch
```

---

## 7) Known Sources of Variability

* BLAS/OpenMP kernels can cause tiny numeric drift.
* Dirichlet splitting (S2) is seed-sensitive; both `alpha` and `seed` matter.
* CIFAR-10 content is stable; randomness affects only client partitioning.

---

## 8) Repro Checklist

* [ ] Use a fixed Python version (see `env/environment.yml`).
* [ ] Set seeds with `frl.utils.set_seed(...)` and `PYTHONHASHSEED`.
* [ ] Use provided scenario YAMLs or add your own under `frl/scenarios/`.
* [ ] Save logs/figures with timestamps.
* [ ] Export `pip freeze`, Python version, and Git commit hash.
