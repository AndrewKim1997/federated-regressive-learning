#!/usr/bin/env bash
set -euo pipefail

# Reproduce minimal runs for S1/S2/S3 with FRL and FedAvg on MNIST.
# Adjust rounds/epochs for speed vs. fidelity.

mkdir -p results/scenarios results/logs results/figures

echo "[1/6] Generate S1 scenario"
python -m frl.scenarios.scenario_gen -c frl/scenarios/s1_equal_dist_diff_size.yaml -o results/scenarios --preview

echo "[2/6] Run FRL on S1"
python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

echo "[3/6] Run FedAvg on S1"
python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
  --aggregator fedavg \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

echo "[4/6] Generate S2 scenario"
python -m frl.scenarios.scenario_gen -c frl/scenarios/s2_hetero_dist_diff_size.yaml -o results/scenarios --preview

echo "[5/6] Run FRL on S2"
python scripts/run_federated.py \
  --dataset mnist \
  --scenario frl/scenarios/s2_hetero_dist_diff_size.yaml \
  --aggregator frl \
  --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

echo "[6/6] Make figures and summary"
python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures

echo "Done. See results/figures and results/logs."
