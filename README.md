# Federated Regressive Learning (FRL)

> Adaptive **server aggregation** that weights client updates by both **data size** and **distribution quality**.
> FRL estimates how close each client’s label distribution is to a chosen **reference** (default: uniform) using a statistical distance (e.g., **Wasserstein**, **JS**), combines it with sample counts, and normalizes to obtain aggregation weights $\beta_i$.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen?logo=githubactions" alt="ci">
  <img src="https://img.shields.io/badge/License-MIT-black" alt="license">
</p>

---

## 🌟 Highlights

* **Method-first reproducibility**: seeded scenario generators (S1/S2/S3) and deterministic pipelines.
* **Drop-in aggregator**: `frl_aggregate(...)` with interpretable weights and metrics.
* **Baselines included**: FedAvg / FedProx wrappers for quick, fair comparisons.
* **Batteries included**: MNIST (NumPy/OpenML), CIFAR-10 (via `torchvision`), figures/tables scripts, CI, and Docker (CPU & CUDA).

---

## 📚 Table of Contents

* [Install](#-install)
* [Quick start](#-quick-start)
* [Scenarios](#-scenarios)
* [Reproducibility](#-reproducibility)
* [Datasets & Privacy](#-datasets--privacy)
* [Docker](#-docker)
* [Repository layout](#-repository-layout)
* [Cite](#-cite)
* [Contributing](#-contributing)
* [License](#-license)

---

## 🚀 Install

### Option A — Conda

```bash
conda env create -f env/environment.yml
conda activate frl
pip install -e .[dev]
```

### Option B — venv + pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

> **Torch note:** FRL’s core runner uses NumPy; `torchvision` is required only for CIFAR-10.

---

## ⚡ Quick start

### 1) Generate a split (S1)

```bash
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s1_equal_dist_diff_size.yaml \
  -o results/scenarios --preview
```

### 2) Run FRL vs baselines (MNIST · CPU)

```bash
python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/make_fig_tables.py \
  --glob "results/logs/*.csv" --outdir results/figures
```

### 3) CIFAR-10 smoke (requires `torchvision`)

```bash
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s2_hetero_dist_diff_size.yaml \
  -o results/scenarios --preview

python scripts/run_federated.py \
  --dataset cifar10 \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 1 --local-epochs 1 --batch-size 256 --lr 0.1 \
  --log-dir results/logs
```

---

## 🧪 Scenarios

* **S1 — equal distribution, different client sizes**
* **S2 — heterogeneous distributions & sizes**

  * Two modes: **Dirichlet** (`alpha`) or **fixed per-client class proportions** (`class_props`).
  * Tip: if you specify `clients.sizes`, make sure the sum equals the dataset size (or leave as `null` to auto-even).
* **S3 — class-missing + heterogeneity**

Edit the YAMLs under `frl/scenarios/` and regenerate; see **`docs/SCENARIOS.md`** for definitions and examples.

---

## 🔁 Reproducibility

* Seeds are set via `frl.utils.set_seed(seed)` and per-scenario YAML `seed`.
* CI (GitHub Actions) runs unit tests and **MNIST/CIFAR smoke**.
* Export provenance with:

```bash
python -m pip freeze > results/requirements-$(date -u +%Y%m%dT%H%M%SZ).txt
python --version   > results/python-version.txt
git rev-parse HEAD > results/git-commit.txt
```

See **`docs/REPRODUCIBILITY.md`** for details.

---

## 🗂️ Datasets & Privacy

* **MNIST**: loaded via scikit-learn (OpenML) to NumPy arrays `[0,1]`.
* **CIFAR-10**: via `torchvision`; converted to NumPy for training.
* **UGEI**: **not distributed** (privacy & contractual restrictions). A placeholder interface is provided; see **`docs/PRIVACY.md`**.

> Never commit raw data. This repo ignores `data/` and large artifacts by default.

---

## 🐳 Docker

Two first-class images are provided:

* **CPU** — no PyTorch/torchvision
  `docker build -f docker/Dockerfile.cpu -t frl:cpu .`
* **CUDA** — includes PyTorch/torchvision (CUDA 12.1 wheels)
  `docker build -f docker/Dockerfile.cuda -t frl:cuda .`

Run (CPU):

```bash
docker run --rm -it -v "$PWD:/app" frl:cpu \
  bash -lc 'python -m frl.scenarios.scenario_gen -c frl/scenarios/s1_equal_dist_diff_size.yaml -o results/scenarios --preview && \
            python scripts/run_federated.py --dataset mnist --scenario frl/scenarios/s1_equal_dist_diff_size.yaml --aggregator frl --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1'
```

Run (CUDA, GPU host):

```bash
docker run --gpus all --rm -it -v "$PWD:/app" frl:cuda \
  bash -lc 'python -m frl.scenarios.scenario_gen -c frl/scenarios/s2_hetero_dist_diff_size.yaml -o results/scenarios --preview && \
            python scripts/run_federated.py --dataset cifar10 --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml --aggregator frl --rounds 1 --local-epochs 1 --batch-size 256 --lr 0.1'
```

See **`docs/DOCKER.md`** for more.

---

## 🧭 Repository layout

```
federated-regressive-learning/
├── frl/                  # library: algorithm, aggregation, metrics, data, scenarios
├── scripts/              # run_federated.py, make_fig_tables.py
├── frl/scenarios/        # S1/S2/S3 YAMLs + generator
├── configs/              # training/optimizer/dataset configs
├── examples/             # minimal, runnable examples
├── docs/                 # REPRODUCIBILITY, SCENARIOS, PRIVACY, DOCKER
├── tests/                # unit & smoke tests
├── env/                  # environment.yml, requirements.txt
└── docker/               # Dockerfile.cpu, Dockerfile.cuda
```

---

## 📝 Cite

If you use this repository, please cite the paper (or see `CITATION.cff`):

```bibtex
@article{kim2024federated,
  title={Federated regressive learning: Adaptive weight updates through statistical information of clients},
  author={Kim, Dong Seok and Ahmad, Shabir and Whangbo, Taeg Keun},
  journal={Applied Soft Computing},
  volume={166},
  pages={112043},
  year={2024},
  publisher={Elsevier}
}
```

---

## 🤝 Contributing

Contributions are welcome!
Please ensure:

* `pytest` passes and **ruff** shows no errors,
* new scenarios/configs include seeds,
* results/logs are excluded from commits.

---

## 📄 License

Code is released under the **MIT License** (see `LICENSE`).
Datasets remain subject to their original licenses; **UGEI is not distributed**.
