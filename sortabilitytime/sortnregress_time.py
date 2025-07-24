"""
Adaptation of code from the CausalDisco repository by Alexander Reisach:
    https://github.com/CausalDisco/CausalDisco

This implementation adapts the non-time-series variable sorting approach (R²-sortability),
as introduced and developed in the following works:

1. Reisach, Alexander, Seiler, Christof, and Weichwald, Sebastian.
   "Beware of the simulated DAG! Causal discovery benchmarks may be easy to game."
   *Advances in Neural Information Processing Systems*, 34:27772–27784, 2021.

2. Reisach, Alexander, Tami, Myriam, Seiler, Christof, Chambaz, Antoine, and Weichwald, Sebastian.
   "A scale-invariant sorting criterion to find a causal order in additive noise models."
   *Advances in Neural Information Processing Systems*, 36:785–807, 2023.

Adapted by:
    Christopher Lohse (lohsecc@tcd.ie)
    During an internship with Professor Jakob Runge's team,
    supervised by Jonas Wahl.
"""

from sklearn.linear_model import LinearRegression, LassoLarsIC
import numpy as np
from scipy.ndimage import shift
from sortabilitytime.sortability_ts import r2coeff, ts_graph_to_summary_graph


def _get_lags(X, tau_max):
    """
    Construct lagged versions of the input time series.

    Args:
        X (np.ndarray): Data of shape (n, d).
        tau_max (int): Maximum lag order.

    Returns:
        np.ndarray: Array of shape (n - tau_max, d, tau_max + 1), where each slice includes
                    the current value and tau_max lags for each variable.
    """
    Xlags = np.array(
        [
            shift(X, shift=[-i, 0], cval=np.nan)[: -tau_max + 1, :]
            for i in range(1, tau_max + 1)
        ]
    )
    T = X.shape[0]
    Xlags = np.moveaxis(Xlags, 0, 1)
    Xlags = np.moveaxis(Xlags, -1, 1)
    X = X[: T - (tau_max), :]
    Xlags = Xlags[: T - (tau_max), :, :]

    XY = np.zeros((Xlags.shape[0], X.shape[1], tau_max + 1))
    XY[:, :, 0] = X
    XY[:, :, 1:] = Xlags

    return XY


def var_sort_regress_ts(X, tau_max=3):
    """
    Time-series causal discovery using variance-based ordering and lags.

    Args:
        X (np.ndarray): Time-series data of shape (n, d).
        tau_max (int): Maximum lag order.

    Returns:
        np.ndarray: Weighted 3D adjacency tensor of shape (d, d, tau_max + 1).
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic", fit_intercept=True)
    XY = _get_lags(X, tau_max=tau_max)
    d = X.shape[1]
    W = np.zeros((d, d, tau_max + 1))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        target = increasing[k]
        indices = increasing[:k]
        x = XY[:, indices, 1:]
        x_shape = x.shape
        x_self = XY[:, k, 1:]
        x = x.reshape((x.shape[0], -1))
        x = np.concatenate((x, x_self), axis=1)
        LR.fit(x, XY[:, target, 0].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(x * weight, XY[:, target, 0].ravel())
        weights = LL.coef_ * weight
        W[indices, target, 1:] = weights[: x_shape[1] * x_shape[2]].reshape(x_shape[1:])
        W[target, target, 1:] = weights[x_shape[1] * x_shape[2] :].reshape(
            1, x_shape[2]
        )
    return W


def r2_sort_regress_ts(X, tau_max=3):
    """
    Time-series causal discovery using R²-based ordering and lags.

    Args:
        X (np.ndarray): Time-series data of shape (n, d).
        tau_max (int): Maximum lag order.

    Returns:
        np.ndarray: Weighted 3D adjacency tensor of shape (d, d, tau_max + 1).
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic", fit_intercept=True)
    XY = _get_lags(X, tau_max=tau_max)
    d = X.shape[1]
    W = np.zeros((d, d, tau_max + 1))
    increasing = np.argsort(r2coeff(X=X.T))

    for k in range(1, d):
        target = increasing[k]
        indices = increasing[:k]
        x = XY[:, indices, 1:]
        x_shape = x.shape
        x_self = XY[:, k, 1:]
        x = x.reshape((x.shape[0], -1))
        x = np.concatenate((x, x_self), axis=1)
        LR.fit(x, XY[:, target, 0].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(x * weight, XY[:, target, 0].ravel())
        weights = LL.coef_ * weight
        W[indices, target, 1:] = weights[: x_shape[1] * x_shape[2]].reshape(x_shape[1:])
        W[target, target, 1:] = weights[x_shape[1] * x_shape[2] :].reshape(
            1, x_shape[2]
        )
    return W


def random_sort_regress_ts(X, tau_max=3, seed=None):
    """
    Time-series causal discovery using a random variable ordering.

    Args:
        X (np.ndarray): Time-series data of shape (n, d).
        tau_max (int): Maximum lag order.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Weighted 3D adjacency tensor of shape (d, d, tau_max + 1).
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic", fit_intercept=True)
    XY = _get_lags(X, tau_max=tau_max)
    d = X.shape[1]
    W = np.zeros((d, d, tau_max + 1))

    if seed is None:
        seed = np.random.randint(0, np.iinfo("int").max)
    rng = np.random.default_rng(seed)

    increasing = rng.permutation(X.shape[1])

    for k in range(1, d):
        target = increasing[k]
        indices = increasing[:k]
        x = XY[:, indices, 1:]
        x_shape = x.shape
        x_self = XY[:, k, 1:]
        x = x.reshape((x.shape[0], -1))
        x = np.concatenate((x, x_self), axis=1)
        LR.fit(x, XY[:, target, 0].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(x * weight, XY[:, target, 0].ravel())
        weights = LL.coef_ * weight
        W[indices, target, 1:] = weights[: x_shape[1] * x_shape[2]].reshape(x_shape[1:])
        W[target, target, 1:] = weights[x_shape[1] * x_shape[2] :].reshape(
            1, x_shape[2]
        )
    return W


def var_sort_regress_ts_reverse(X, tau_max=3):
    """
    Time-series causal discovery using reverse variance ordering.

    Args:
        X (np.ndarray): Time-series data of shape (n, d).
        tau_max (int): Maximum lag order.

    Returns:
        np.ndarray: Weighted 3D adjacency tensor of shape (d, d, tau_max + 1).
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic", fit_intercept=True)
    XY = _get_lags(X, tau_max=tau_max)
    d = X.shape[1]
    W = np.zeros((d, d, tau_max + 1))
    increasing = np.argsort(np.var(X, axis=0))[::-1]

    for k in range(1, d):
        target = increasing[k]
        indices = increasing[:k]
        x = XY[:, indices, 1:]
        x_shape = x.shape
        x_self = XY[:, k, 1:]
        x = x.reshape((x.shape[0], -1))
        x = np.concatenate((x, x_self), axis=1)
        LR.fit(x, XY[:, target, 0].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(x * weight, XY[:, target, 0].ravel())
        weights = LL.coef_ * weight
        W[indices, target, 1:] = weights[: x_shape[1] * x_shape[2]].reshape(x_shape[1:])
        W[target, target, 1:] = weights[x_shape[1] * x_shape[2] :].reshape(
            1, x_shape[2]
        )
    return W


def sort_regress(X, scores):
    """
    Regress each variable onto all predecessors in
    the ordering implied by the scores.

    Args:
        X: Data (:math:`n \times d` np.array).
        scores: Vector of scores (np.array with :math:`d` entries).

    Returns:
        Candidate causal structure matrix with coefficients
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic", max_iter=50)
    d = X.shape[1]
    W = np.zeros((d, d))
    ordering = np.argsort(scores)

    # backward regression
    for k in range(1, d):
        cov = ordering[:k]
        target = ordering[k]
        LR.fit(X[:, cov], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, cov] * weight, X[:, target].ravel())
        W[cov, target] = LL.coef_ * weight
    return W


def r2_sort_regress(X):
    r"""
    Perform sort_regress using :math:`R^2` as ordering criterion.

    Args:
        X: Data (:math:`n \times d` np.array).

    Returns:
        Candidate causal structure matrix with coefficients.
    """
    return sort_regress(X, r2coeff(X.T))


def var_sort_regress(X):
    """Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates."""
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight

    return W


def var_sort_regress_reverse(X):
    """Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates."""
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))[::-1]

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight

    return W


def random_sort_regress(X, seed=None):
    """
    Perform sort_regress using a random order.

    Args:
        X: Data (:math:`n \times d` np.array).
        seed (optional): random seed (integer)

    Returns:
        Candidate causal structure matrix with coefficients.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo("int").max)
    rng = np.random.default_rng(seed)
    return sort_regress(X, rng.permutation(X.shape[1]))


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    import numpy as np

    # Example usage
    np.random.seed(0)
    d = 5
    n = 100
    tau_max = 3

    # Generate random data (n samples, d variables)
    X = np.random.randn(n, d)

    # Generate random ground-truth DAG adjacency tensor (d x d x tau_max)
    W_true = np.random.rand(d, d, tau_max) < 0.2

    # Convert to summary graph (d x d), assuming this function exists
    W_summary = ts_graph_to_summary_graph(W_true)

    # Helper function to binarize weights for F1 score
    def binarize(W, threshold=1e-5):
        return (np.abs(W) > threshold).astype(int)

    # Get predictions from different sortnregress_ts methods
    W_r2 = r2_sort_regress_ts(X, tau_max)
    W_var = var_sort_regress_ts(X, tau_max)
    W_rand = random_sort_regress_ts(X, tau_max, seed=42)

    # Convert to summary graphs and binarize for comparison
    W_r2_sum = binarize(ts_graph_to_summary_graph(W_r2))
    W_var_sum = binarize(ts_graph_to_summary_graph(W_var))
    W_rand_sum = binarize(ts_graph_to_summary_graph(W_rand))
    W_true_bin = binarize(W_summary)

    # Flatten matrices for F1 score calculation
    y_true = W_true_bin.flatten()
    y_r2 = W_r2_sum.flatten()
    y_var = W_var_sum.flatten()
    y_rand = W_rand_sum.flatten()

    # Compute and print F1 scores
    print("F1 score (R^2 sortnregress):", f1_score(y_true, y_r2))
    print("F1 score (Variance sortnregress):", f1_score(y_true, y_var))
    print("F1 score (Random sortnregress):", f1_score(y_true, y_rand))
