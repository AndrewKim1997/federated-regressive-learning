import numpy as np
from frl.aggregation import ClientUpdate
from frl.algo_frl import compute_frl_weights, frl_aggregate


def _updates_basic(D=4):
    """Three clients with symmetric distributions around uniform."""
    u0 = ClientUpdate(delta=np.ones(D, dtype=np.float32),  num_samples=100, dist=np.array([0.9, 0.1]))
    u1 = ClientUpdate(delta=np.zeros(D, dtype=np.float32), num_samples=100, dist=np.array([0.5, 0.5]))
    u2 = ClientUpdate(delta=-np.ones(D, dtype=np.float32), num_samples=100, dist=np.array([0.1, 0.9]))
    return [u0, u1, u2]


def test_compute_frl_weights_wasserstein_prefers_close_to_ref():
    ups = _updates_basic()
    beta, info = compute_frl_weights(ups, ref="uniform", metric="wasserstein", size_power=1.0, dist_power=1.0)
    # Client with uniform distribution should receive the largest weight
    assert np.isclose(beta.sum(), 1.0)
    assert beta[1] > beta[0] and beta[1] > beta[2]


def test_compute_frl_weights_js_metric_also_valid():
    ups = _updates_basic()
    beta, _ = compute_frl_weights(ups, ref="uniform", metric="js")
    assert np.isclose(beta.sum(), 1.0)
    assert beta[1] > max(beta[0], beta[2])


def test_frl_aggregate_matches_manual_weighted_average():
    ups = _updates_basic(D=6)
    delta, info = frl_aggregate(ups, ref="uniform", metric="wasserstein")
    # Reconstruct by manual weighted sum over flat vectors
    W = info["weights"]
    stacked = np.stack([u.delta.reshape(-1) for u in ups], axis=0)
    expected = (W[:, None] * stacked).sum(axis=0)
    assert delta.shape == ups[0].delta.shape
    assert np.allclose(delta.reshape(-1), expected, atol=1e-6)
