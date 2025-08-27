# Docker Usage

This repository ships two first-class Docker images:

* **CPU image** – lightweight, no PyTorch/torchvision required. Great for core logic, MNIST (S1), docs, and CI.
* **CUDA image** – includes PyTorch/torchvision (CUDA 12.1 wheels). Use on a machine with an NVIDIA driver and `nvidia-container-toolkit`.

Both images are defined under:

* `docker/Dockerfile.cpu`
* `docker/Dockerfile.cuda`

A `.dockerignore` is provided to keep images small.

---

## 1) Requirements

* **Docker 20.10+**
* For GPU runs:

  * Host **NVIDIA driver** installed (no CUDA toolkit needed on host)
  * **nvidia-container-toolkit** installed and configured
    (verify with `docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi`)

---

## 2) Build locally (optional)

```bash
# From the repository root:

# CPU image
docker build -f docker/Dockerfile.cpu -t frl:cpu .

# CUDA image (PyTorch CUDA 12.1 wheels)
docker build -f docker/Dockerfile.cuda -t frl:cuda .
```

> Tip: add `--pull` occasionally to refresh base layers.

---

## 3) Pull from GitHub Container Registry (optional)

If you use the provided GitHub Actions workflow (`.github/workflows/docker.yml`), images are pushed to GHCR:

```
ghcr.io/<OWNER>/<REPO>:cpu-latest
ghcr.io/<OWNER>/<REPO>:cuda-latest
```

You can also pin to an immutable tag, e.g. `:cpu-sha-<abcdef0>` or a release tag `:cuda-v0.1.0`.

```bash
docker pull ghcr.io/<OWNER>/<REPO>:cpu-latest
docker pull ghcr.io/<OWNER>/<REPO>:cuda-latest
```

---

## 4) Run examples

Mount the repo into `/app` so results land under your working tree (`results/`, `data/`).

### 4.1 CPU · MNIST S1 (no PyTorch required)

```bash
docker run --rm -it -v "$PWD:/app" frl:cpu \
  bash -lc '
    python -m frl.scenarios.scenario_gen \
      -c frl/scenarios/s1_equal_dist_diff_size.yaml \
      -o results/scenarios --preview && \
    python scripts/run_federated.py \
      --dataset mnist \
      --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
      --aggregator frl \
      --rounds 1 --local-epochs 1 --batch-size 256 --lr 0.1 \
      --log-dir results/logs && \
    python scripts/make_fig_tables.py \
      --glob "results/logs/*.csv" --outdir results/figures'
```

### 4.2 CUDA · CIFAR-10 S2 (requires GPU)

```bash
docker run --gpus all --rm -it -v "$PWD:/app" \
  -e TORCH_HOME=/app/.torch \
  frl:cuda \
  bash -lc '
    python -m frl.scenarios.scenario_gen \
      -c frl/scenarios/s2_hetero_dist_diff_size.yaml \
      -o results/scenarios --preview && \
    python scripts/run_federated.py \
      --dataset cifar10 \
      --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
      --aggregator frl \
      --rounds 1 --local-epochs 1 --batch-size 256 --lr 0.1 \
      --log-dir results/logs'
```

> The CUDA image prints `torch.cuda.is_available()` on start; you can also test inside the container:
>
> ```bash
> python -c "import torch; print('CUDA:', torch.cuda.is_available())"
> ```

---

## 5) Useful environment & flags

* **Thread caps for deterministic runs** (already set in CI; you can pass them on run if needed):

  ```bash
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONHASHSEED=42
  ```
* **Cache datasets/weights**:

  * CPU image uses NumPy loaders; CIFAR-10 via `torchvision` (CUDA image).
  * Set `-e TORCH_HOME=/app/.torch` to cache under your repo (mounted).
* **File permissions**:

  * Images run as a non-root `appuser` (UID 1000). If you see permission issues on some hosts, mount with `:z` (SELinux) or adjust directory ownership on the host.

---

## 6) Reproducibility tips

* **Pin image tags** in your scripts (avoid `:latest` for papers).
* Keep `configs/` and `frl/scenarios/*.yaml` under version control; logs/figures go to `results/` (ignored by Git).
* Export runtime metadata during runs:

  ```bash
  python -m pip freeze > results/requirements-$(date -u +%Y%m%dT%H%M%SZ).txt
  git rev-parse HEAD > results/git-commit.txt
  ```

---

## 7) Troubleshooting

* **`CUDA: False` inside CUDA image**

  * Check host driver: `nvidia-smi` on host.
  * Ensure `--gpus all` and `nvidia-container-toolkit` are configured.
  * Host driver must support the CUDA runtime used by the container (CUDA 12.1 in this image).

* **CIFAR-10 load error (`torchvision` missing)**

  * Use the **CUDA** image (it installs `torch`/`torchvision`), or install `torchvision` in a custom CPU image if you need CIFAR-10 without GPU.

* **Slow builds / huge images**

  * `.dockerignore` is included—keep `data/`, `results/` out of the context.
  * Use fixed base images and avoid unnecessary system packages.

---

## 8) Extending images

* **Multi-arch (amd64/arm64)**: enable in the GitHub Actions `docker.yml` by setting `platforms: linux/amd64,linux/arm64` on the build step (requires QEMU).
* **Extra dependencies**: fork `Dockerfile.*` and add `pip install ...` blocks; prefer pinning versions for paper artifacts.
* **GPU variants**: switch to a CUDA runtime base image if you need system-level CUDA libraries; our current approach uses PyTorch wheels (`cu121`) which bundle CUDA.

---

## 9) Quick reference

* Build CPU: `docker build -f docker/Dockerfile.cpu -t frl:cpu .`
* Build CUDA: `docker build -f docker/Dockerfile.cuda -t frl:cuda .`
* Run MNIST (CPU): `docker run --rm -it -v "$PWD:/app" frl:cpu ...`
* Run CIFAR-10 (GPU): `docker run --gpus all --rm -it -v "$PWD:/app" frl:cuda ...`
* Pull from GHCR: `docker pull ghcr.io/<OWNER>/<REPO>:cpu-latest` (or `:cuda-latest`)

If you need a **build-and-push** pipeline, see `.github/workflows/docker.yml` which builds both flavors and publishes them to GHCR.
