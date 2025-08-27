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

from .aggregation import ClientUpdate, weighted_average
from .baselines import fedavg_aggregate, fedprox_aggregate
from .algo_frl import frl_aggregate
from .metrics import (
    accuracy,
    precision_recall_f1,
    ece,
    js_divergence,
    wasserstein_discrete,
)
from .utils import set_seed, get_logger

__all__ = [
    "ClientUpdate",
    "weighted_average",
    "fedavg_aggregate",
    "fedprox_aggregate",
    "frl_aggregate",
    "accuracy",
    "precision_recall_f1",
    "ece",
    "js_divergence",
    "wasserstein_discrete",
    "set_seed",
    "get_logger",
]

__version__ = "0.1.0"
