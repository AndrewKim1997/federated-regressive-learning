#!/usr/bin/env bash
set -euo pipefail

# S1: equal distribution, different client sizes (MNIST)
python -m frl.scenarios.scenario_gen \
  -c frl/scenarios/s1_equal_dist_diff_size.yaml \
  -o results/scenarios --preview

python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator fedavg \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures
echo "Done. See results/logs and results/figures."
