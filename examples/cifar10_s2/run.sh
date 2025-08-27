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
