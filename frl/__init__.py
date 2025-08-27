"""
Federated Regressive Learning (FRL) – method-reproduction package.

Public API:
    - aggregation.ClientUpdate
    - aggregation.weighted_average
    - baselines.fedavg_aggregate, baselines.fedprox_aggregate
    - algo_frl.frl_aggregate
    - metrics.accuracy, metrics.precision_recall_f1, metrics.ece, metrics.js_divergence, metrics.wasserstein_discrete
    - utils.set_seed, utils.get_logger
"""

# frl/__init__.py
from .algo_frl import compute_frl_weights, frl_aggregate
from .aggregation import fedavg_aggregate, fedprox_aggregate, ClientUpdate
from .metrics import accuracy, ece, js_divergence, precision_recall_f1
from .utils import set_seed

__all__ = [
    "ClientUpdate",
    "compute_frl_weights",
    "frl_aggregate",
    "fedavg_aggregate",
    "fedprox_aggregate",
    "accuracy",
    "ece",
    "js_divergence",
    "precision_recall_f1",
    "set_seed",
]
__version__ = "0.1.0"
