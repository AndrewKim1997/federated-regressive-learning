<!-- examples/mnist_s1/README.md -->
# MNIST · Scenario S1 (equal distribution, different sizes)

This example shows how to:
1) generate S1 scenario indices,
2) run FRL and FedAvg on MNIST with a small number of rounds,
3) produce simple plots and a summary table.

## Quick start

    # 1) Generate scenario (indices + summary CSV)
    python -m frl.scenarios.scenario_gen \
      -c frl/scenarios/s1_equal_dist_diff_size.yaml \
      -o results/scenarios --preview

    # 2) Run FRL
    python scripts/run_federated.py \
      --dataset mnist \
      --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
      --aggregator frl \
      --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

    # 3) Run FedAvg
    python scripts/run_federated.py \
      --dataset mnist \
      --scenario frl/scenarios/s1_equal_dist_diff_size.yaml \
      --aggregator fedavg \
      --rounds 3 --local-epochs 1 --batch-size 256 --lr 0.1

    # 4) Make figures and a final summary
    python scripts/make_fig_tables.py --glob "results/logs/*.csv" --outdir results/figures

## Inspect FRL weights (one-shot)

    python examples/mnist_s1/inspect_weights.py

This prints per-client class distributions and the FRL weights computed against the chosen reference distribution.
