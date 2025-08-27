# Reproducibility Guide

This repository focuses on **method reproducibility** for Federated Regressive Learning (FRL). Exact experimental conditions from the 2024 paper are not fully recoverable; instead, we provide **seeded, programmatic scenarios** and **deterministic pipelines** to reproduce trends.

---

## 1) Environment

### Option A: Conda
```bash
conda env create -f env/environment.yml
conda activate frl
# (optional) Install PyTorch CPU wheels if you plan to use torchvision-based loaders:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
