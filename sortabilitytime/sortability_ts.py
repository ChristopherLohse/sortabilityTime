"""
Adaptation of code from the CausalDisco repository by Alexander Reisach:
    https://github.com/CausalDisco/CausalDisco

This implementation adapts the non-time-series variable sorting approach (R²-sortability),
as introduced and developed in the following works:

1. Reisach, Alexander, Seiler, Christof, and Weichwald, Sebastian.
   "Beware of the simulated DAG! Causal discovery benchmarks may be easy to game."
   *Advances in Neural Information Processing Systems*, 34:27772–27784, 2021.

   @article{reisach2021beware,
       title   = {Beware of the simulated dag! causal discovery benchmarks may be easy to game},
       author  = {Reisach, Alexander and Seiler, Christof and Weichwald, Sebastian},
       journal = {Advances in Neural Information Processing Systems},
       volume  = {34},
       pages   = {27772--27784},
       year    = {2021},
   }

2. Reisach, Alexander, Tami, Myriam, Seiler, Christof, Chambaz, Antoine, and Weichwald, Sebastian.
   "A scale-invariant sorting criterion to find a causal order in additive noise models."
   *Advances in Neural Information Processing Systems*, 36:785–807, 2023.

   @article{reisach2023scale,
       title   = {A scale-invariant sorting criterion to find a causal order in additive noise models},
       author  = {Reisach, Alexander and Tami, Myriam and Seiler, Christof and Chambaz, Antoine and Weichwald, Sebastian},
       journal = {Advances in Neural Information Processing Systems},
       volume  = {36},
       pages   = {785--807},
       year    = {2023},
   }

Adapted by:
    Christopher Lohse (lohsec@tcd.ie)
    During an internship with Professor Jakob Runge's team,
    supervised by Jonas Wahl.
"""

import itertools
from sklearn.linear_model import LinearRegression
from scipy import linalg
import numpy as np
import networkx as nx


# main contribution of paper
def __order_alignment(W, scores, tol=0.0):
    r"""
    Computes a measure of agreement between a causal ordering—implied by the topology of
    the (weighted) adjacency matrix `W`—and an ordering induced by the given `scores`.

    Corrects for any cycles that may occur in the time series summary graph.

    Args:
        W (np.ndarray): Weighted or binary DAG adjacency matrix of shape (d, d).
        scores (np.ndarray): Vector of scores with d entries, one per variable.
        tol (float, optional): Tolerance threshold for score comparisons (non-negative).

    Returns:
        float: Scalar value in [0, 1] measuring the alignment between graph topology and score-based ordering.
    """

    assert tol >= 0.0, "tol must be non-negative"
    E = W != 0
    Ek = E.copy()
    n_paths = 0
    n_correctly_ordered_paths = 0

    # arrange scores as row vector
    scores = scores.reshape(1, -1)
    G = nx.DiGraph(incoming_graph_data=E)
    cycles = []
    # main contribution of the paper implements
    #     #\begin{equation}
    #     s := \frac{\sum_{(i,j)\in \mathcal{AP}(\mathcal{G}_{\mathrm{sum}}) }  increasing(cri(X_i), cri(X_j))}{\sum_{(i,j)\in \mathcal{AP}(\mathcal{G}_{\mathrm{sum}}) } 1} \in [0, 1], \label{eq:sor_mod}
    # \end{equation}

    for comb in itertools.combinations(G.nodes, 2):
        if (
            nx.has_path(G, *comb)
            and nx.has_path(G, comb[1], comb[0])
            and nx.shortest_path_length(G, *comb) < len(E) - 1
        ):
            cycles.append((comb[0], comb[1]))
    # create d x d matrix of score differences of scores such that
    # the entry in the i-th row and j-th column is
    #     * positive if score i < score j
    #     * zero if score i = score j
    #     * negative if score i > score j
    differences = scores - scores.T

    # measure ordering agreement
    # see 10.48550/arXiv.2102.13647, Section 3.1
    # and 10.48550/arXiv.2303.18211, Equation (3)
    for _ in range(len(E) - 1):
        n_paths += Ek.sum()
        # count 1/2 per correctly ordered or unordered pair
        n_correctly_ordered_paths += (Ek * (differences >= 0 - tol)).sum() / 2
        # count another 1/2 per correctly ordered pair
        n_correctly_ordered_paths += (Ek * (differences > 0 + tol)).sum() / 2

        Ek = Ek.dot(E)
    if n_paths == 0:
        return np.nan

    n_correctly_ordered_paths -= len(cycles)
    n_paths -= len(cycles) * 2

    return n_correctly_ordered_paths / n_paths


def r2coeff(X):
    r"""
    Compute the :math:`R^2` of each variable using partial correlations obtained through matrix inversion.

    Args:
        X: Data (:math:`d \times n` np.array - note that the dimensions here are different from other methods, following np.corrcoef).

    Returns:
        Array of :math:`R^2` values for all variables.
    """
    try:
        return 1 - 1 / np.diag(linalg.inv(np.corrcoef(X)))
    except:
        print("fallback")
        # fallback if correlation matrix is singular
        d = X.shape[0]
        r2s = np.zeros(d)
        LR = LinearRegression()
        X = X.T
        for k in range(d):
            parents = np.arange(d) != k
            LR.fit(X[:, parents], X[:, k])
            r2s[k] = LR.score(X[:, parents], X[:, k])
        return r2s


def var_sortability(X, W, tol=0.0):
    r"""
    Sortability by variance.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).

    Returns:
        Var-sortability value (:math:`\in [0, 1]`) of the data
    """
    return __order_alignment(W, np.nanvar(X, axis=0), tol=tol)


def r2_sortability(X, W, tol=0.0):
    r"""
    Sortability by :math:`R^2`.

    Args:
        X: Data (:math:`n \times d` np.array).
        W: Weighted/Binary ground-truth DAG adjacency matrix (:math:`d \times d` np.array).

    Returns:
        :math:`R^2`-sortability value (:math:`\in [0, 1]`) of the data
    """
    return __order_alignment(W, r2coeff(X.T), tol=tol)


def ts_graph_to_summary_graph(ts_graph: np.array) -> np.array:
    r"""
    Compute the summary graph of a time series graph.

    Args:
        ts_graph: Time series graph (:math:`d \times d \times T` np.array).

    Returns:
        Summary graph (:math:`d \times d` np.array).
    """
    sum_adj = np.sum(ts_graph, axis=2)

    sum_adj[sum_adj != 0] = 1
    np.fill_diagonal(sum_adj, 0)

    return sum_adj


if __name__ == "__main__":
    # Example usage
    # Generate random data
    np.random.seed(0)
    d = 5
    n = 100
    X = np.random.randn(n, d)
    tau_max = 3

    # Generate random ground-truth DAG
    W = np.random.rand(d, d, tau_max) < 0.2

    W = ts_graph_to_summary_graph(W)

    # Compute sortability by variance
    print(var_sortability(X, W))

    # Compute sortability by R^2
    print(r2_sortability(X, W))
