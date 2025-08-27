# CIFAR-10 · Scenario S2 (heterogeneous distributions, different sizes)

This example demonstrates a non-IID setting on CIFAR-10 with S2.
**Requires `torchvision`** to download and load CIFAR-10.

## Quick start
```bash
# Install torchvision if not present
pip install torchvision  # or CPU wheels from the PyTorch index

# 1) Generate S2 scenario (Dirichlet by default, alpha=0.3)
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s2_hetero_dist_diff_size.yaml \
  -o results/scenarios --preview

# 2) Run FRL
python scripts/run_federated.py \
  --dataset cifar10 \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

# 3) Run FedAvg
python scripts/run_federated.py \
  --dataset cifar10 \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator fedavg \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

# 4) Make figures and a final summary
python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures
````

## Notes

* S2 can be generated either via a **Dirichlet splitter** (`alpha`) or **fixed per-client class proportions** (`class_props`).
* Edit `frl/scenarios/s2_hetero_dist_diff_size.yaml` to switch between the two modes.

````

---

#### `examples/cifar10_s2/run.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

# Ensure torchvision is available for CIFAR-10
python - <<'PY'
try:
    import torchvision  # noqa
    print("torchvision OK")
except Exception:
    raise SystemExit("torchvision is required: pip install torchvision")
PY

# S2: heterogeneous distributions, equal sizes by default (CIFAR-10)
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s2_hetero_dist_diff_size.yaml \
  -o results/scenarios --preview

python scripts/run_federated.py \
  --dataset cifar10 \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/run_federated.py \
  --dataset cifar10 \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator fedavg \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures
echo "Done. See results/logs and results/figures."
````
