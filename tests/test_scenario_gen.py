import numpy as np
from frl.data.transforms import class_distribution
from frl.metrics import js_divergence
from frl.scenarios.scenario_gen import build_s1, build_s2, build_s3


def _make_labels(n=1000, num_classes=10, seed=0):
    """Create a balanced label vector of length n over num_classes."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(n // num_classes, c) for c in range(num_classes)])
    # If n not divisible, pad with random classes
    if len(y) < n:
        y = np.concatenate([y, rng.integers(0, num_classes, size=n - len(y))])
    rng.shuffle(y)
    return y.astype(int)


def test_s1_equal_dist_diff_size():
    y = _make_labels(n=1200, num_classes=10, seed=1)
    K = 10
    sizes = [500, 400, 300]
    splits = build_s1(y=y, K=K, num_clients=3, sizes=sizes, seed=42)

    # size check
    assert [len(ix) for ix in splits] == sizes

    # distribution check: per-client distribution close to global distribution
    p_global = class_distribution(y, num_classes=K)
    for ix in splits:
        p = class_distribution(y[ix], num_classes=K)
        # allow small tolerance due to integer rounding
        assert np.allclose(p, p_global, atol=0.05)


def test_s2_dirichlet_has_heterogeneity():
    y = _make_labels(n=3000, num_classes=10, seed=2)
    K = 10
    sizes = [1000, 1000, 1000]
    splits = build_s2(y=y, K=K, num_clients=3, sizes=sizes, seed=7, alpha=0.1)

    dists = [class_distribution(y[ix], num_classes=K) for ix in splits]
    # At least two clients should differ noticeably (mean JS divergence > threshold)
    pair_js = []
    for i in range(3):
        for j in range(i + 1, 3):
            pair_js.append(js_divergence(dists[i], dists[j]))
    assert np.mean(pair_js) > 0.05


def test_s3_class_missing_removes_classes():
    y = _make_labels(n=1500, num_classes=10, seed=3)
    K = 10
    sizes = [500, 500, 500]
    missing_map = {0: [0, 1], 1: [2], 2: []}
    splits = build_s3(
        y=y,
        K=K,
        num_clients=3,
        sizes=sizes,
        seed=11,
        base="iid",
        alpha=0.5,
        missing_map=missing_map,
        fill_to_target=False,
    )

    # Client 0 should have no labels 0 or 1; client 1 no label 2
    y0 = y[splits[0]]
    y1 = y[splits[1]]
    assert not np.any(np.isin(y0, [0, 1]))
    assert not np.any(y1 == 2)
