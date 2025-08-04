import numpy as np
from sklearn.metrics import f1_score
from sortabilitytime.sortability_ts import (
    ts_graph_to_summary_graph,
    var_sortability,
    r2_sortability,
)
from sortabilitytime.sortnregress_time import (
    ts_graph_to_summary_graph,
    r2_sort_regress_ts,
    var_sort_regress_ts,
    random_sort_regress_ts,
)


def test_sortability_measures():
    np.random.seed(0)
    d = 5
    n = 100
    X = np.random.randn(n, d)
    tau_max = 3

    W = np.random.rand(d, d, tau_max) < 0.2
    W_summary = ts_graph_to_summary_graph(W)

    var_score = var_sortability(X, W_summary)
    r2_score = r2_sortability(X, W_summary)

    assert 0 <= var_score <= 1
    assert 0 <= r2_score <= 1
    assert isinstance(var_score, float) and not np.isnan(var_score)
    assert isinstance(r2_score, float) and not np.isnan(r2_score)


def binarize(W, threshold=1e-5):
    return (np.abs(W) > threshold).astype(int)


def test_sortnregress_f1_scores():
    np.random.seed(0)
    d = 5
    n = 100
    tau_max = 3

    X = np.random.randn(n, d)
    W_true = np.random.rand(d, d, tau_max) < 0.2
    W_true_summary = ts_graph_to_summary_graph(W_true)
    W_true_bin = binarize(W_true_summary)

    # Run sortnregress_ts methods
    W_r2 = r2_sort_regress_ts(X, tau_max)
    W_var = var_sort_regress_ts(X, tau_max)
    W_rand = random_sort_regress_ts(X, tau_max, seed=42)

    # Binarize predictions
    y_true = W_true_bin.flatten()
    y_r2 = binarize(ts_graph_to_summary_graph(W_r2)).flatten()
    y_var = binarize(ts_graph_to_summary_graph(W_var)).flatten()
    y_rand = binarize(ts_graph_to_summary_graph(W_rand)).flatten()

    # Compute F1 scores
    f1_r2 = f1_score(y_true, y_r2)
    f1_var = f1_score(y_true, y_var)
    f1_rand = f1_score(y_true, y_rand)

    # Basic assertions
    assert 0 <= f1_r2 <= 1, f"F1 R2 out of range: {f1_r2}"
    assert 0 <= f1_var <= 1, f"F1 VAR out of range: {f1_var}"
    assert 0 <= f1_rand <= 1, f"F1 RAND out of range: {f1_rand}"
