"""
Adaptation of the DYNOTEARS Paper:
    DYNOTEARS: Structure Learning from Time-Series Data
    https://arxiv.org/abs/2002.00498

    @inproceedings{pamfil2020dynotears,
        title     = {DYNOTEARS: Structure Learning from Time-Series Data},
        author    = {Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and
                     Pilgerstorfer, Philip and Georgatzis, Konstantinos and
                     Beaumont, Paul and Aragam, Bryon},
        booktitle = {International Conference on Artificial Intelligence and Statistics},
        pages     = {1595--1605},
        year      = {2020},
    }

This implementation adapts the DYNOTEARS method to integrate with the Tigramite library:
    https://github.com/jakobrunge/tigramite

Adapted by:
    Christopher Lohse (lohsec@tcd.ie)
    During an internship with Professor Jakob Runge's team,
    supervised by Jonas Wahl.
"""

import tigramite.data_processing as pp
from typing import List, Tuple
import numpy as np
import scipy.linalg as slin
from scipy.ndimage import shift
import scipy.optimize as sopt
import warnings

from tigramite import plotting as tp
import matplotlib.pyplot as plt


class Dynotears:
    def __init__(
        self,
        dataframe: pp.DataFrame,
        verbosity: int = 1,
    ) -> None:
        self.dataframe = dataframe
        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity
        self.var_names = self.dataframe.var_names
        # Store the shape of the data in the T and N variables
        self.T = self.dataframe.T
        self.N = self.dataframe.N
        self.history = {"loss": [], "graphs": []}

    def _learn_dynamic_structure(
        self,
        X: np.ndarray,
        Xlags: np.ndarray,
        bnds: List[Tuple[float, float]],
        lambda_w: float = 0.1,
        lambda_a: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.size == 0:
            raise ValueError("Input data X is empty, cannot learn any structure")
        if Xlags.size == 0:
            raise ValueError("Input data Xlags is empty, cannot learn any structure")
        if X.shape[0] != Xlags.shape[0]:
            raise ValueError("Input data X and Xlags must have the same number of rows")
        if Xlags.shape[1] % X.shape[1] != 0:
            raise ValueError(
                "Number of columns of Xlags must be a multiple of number of columns of X"
            )

        _, d_vars = X.shape
        N = self.N
        p_orders = Xlags.shape[1] // d_vars

        def _reshape_wa(
            wa_vec: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Helper function for `_learn_dynamic_structure`. Transform adjacency vector to matrix form

            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
                N (int): number of variables in the model
                tau_max (int): number of past indexes we to use
            Returns:
                intra- and inter-slice adjacency matrices
            """

            w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
            w_plus = w_tilde[:d_vars, :]
            w_minus = w_tilde[d_vars : 2 * d_vars, :]
            w_mat = w_plus - w_minus
            a_plus = (
                w_tilde[2 * d_vars :]
                .reshape(2 * p_orders, d_vars**2)[::2]
                .reshape(d_vars * p_orders, d_vars)
            )
            a_minus = (
                w_tilde[2 * d_vars :]
                .reshape(2 * p_orders, d_vars**2)[1::2]
                .reshape(d_vars * p_orders, d_vars)
            )
            a_mat = a_plus - a_minus
            return w_mat, a_mat

        def _obj_func(wa_vec: np.ndarray) -> float:
            """
            Objective function that dynotears tries to minimise

            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

            Returns:
                float: objective
            """
            n, _ = X.shape
            _w_mat, _a_mat = _reshape_wa(wa_vec)
            loss = (
                0.5
                / n
                * np.square(
                    np.linalg.norm(
                        X.dot(np.eye(N, N) - _w_mat) - Xlags.dot(_a_mat),
                        "fro",
                    )
                )
            )
            _h_value = _h(wa_vec)
            l1_penalty = lambda_w * (wa_vec[: 2 * N**2].sum()) + lambda_a * (
                wa_vec[2 * N**2 :].sum()
            )
            return (
                loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty,
            )

        def _h(wa_vec: np.ndarray) -> float:
            """
            Constraint function of the dynotears

            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

            Returns:
                float: DAGness of the intra-slice adjacency matrix W (0 == DAG, >0 == cyclic)
            """

            _w_mat, _ = _reshape_wa(wa_vec)
            return np.trace(slin.expm(_w_mat * _w_mat)) - N

        def _grad(wa_vec: np.ndarray) -> np.ndarray:
            """
            Gradient function used to compute next step in dynotears

            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

            Returns:
                gradient vector
            """
            n, _ = X.shape
            _w_mat, _a_mat = _reshape_wa(wa_vec)
            e_mat = slin.expm(_w_mat * _w_mat)
            loss_grad_w = (
                -1.0
                / n
                * (
                    X.T.dot(
                        X.dot(np.eye(N, N) - _w_mat) - Xlags.dot(_a_mat),
                    )
                )
            )
            obj_grad_w = (
                loss_grad_w
                + (rho * (np.trace(e_mat) - N) + alpha) * e_mat.T * _w_mat * 2
            )
            obj_grad_a = (
                -1.0
                / n
                * (Xlags.T.dot(X.dot(np.eye(N, N) - _w_mat) - Xlags.dot(_a_mat)))
            )

            grad_vec_w = np.append(
                obj_grad_w, -obj_grad_w, axis=0
            ).flatten() + lambda_w * np.ones(2 * N**2)
            grad_vec_a = obj_grad_a.reshape(p_orders, N**2)
            grad_vec_a = np.hstack(
                (grad_vec_a, -grad_vec_a)
            ).flatten() + lambda_a * np.ones(2 * (p_orders) * N**2)
            return np.append(grad_vec_w, grad_vec_a, axis=0)

        # initialise matrix, weights and constraints
        wa_est = np.zeros(2 * (p_orders + 1) * self.N**2)
        wa_new = np.zeros(2 * (p_orders + 1) * self.N**2)
        rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf
        for n_iter in range(max_iter):
            while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
                wa_new = sopt.minimize(
                    _obj_func,
                    wa_est,
                    method="L-BFGS-B",
                    jac=_grad,
                    bounds=bnds,
                ).x
                h_new = _h(wa_new)
                self.history["loss"].append(h_new)
                self.history["graphs"].append(_reshape_wa(wa_new))
                if h_new > 0.25 * h_value:
                    rho *= 10

            wa_est = wa_new
            h_value = h_new
            alpha += rho * h_value
            if h_value <= h_tol:
                break
            if h_value > h_tol and n_iter == max_iter - 1:
                warnings.warn("Failed to converge. Consider increasing max_iter.")
        return _reshape_wa(wa_est)

    def _get_lags(self):
        X = self.dataframe.values[0].copy()
        #  Xlags (np.ndarray): shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        Xlags = np.array(
            [
                shift(X, shift=[-i, 0], cval=np.nan)[: -self.tau_max, :]
                for i in range(1, self.tau_max)
            ]
        )
        Xlags = np.moveaxis(Xlags, 0, 1)
        Xlags = Xlags.reshape(Xlags.shape[0], Xlags.shape[1] * Xlags.shape[2])
        X = X[: -self.tau_max, :]

        self.X = X
        self.Xlags = Xlags
        return X, Xlags

    def _get_bounds(self, tabu_edges, tabu_parent_nodes, tabu_child_nodes, X, Xlags):
        bnds_w = 2 * [
            (
                (0, 0)
                if i == j
                else (
                    (0, 0)
                    if tabu_edges is not None and (0, i, j) in tabu_edges
                    else (
                        (0, 0)
                        if tabu_parent_nodes is not None and i in tabu_parent_nodes
                        else (
                            (0, 0)
                            if tabu_child_nodes is not None and j in tabu_child_nodes
                            else (0, None)
                        )
                    )
                )
            )
            for i in range(self.N)
            for j in range(self.N)
        ]

        bnds_a = []
        n, d_vars = X.shape
        p_orders = Xlags.shape[1] // d_vars
        for k in range(1, p_orders + 1):
            bnds_a.extend(
                2
                * [
                    (
                        (0, 0)
                        if tabu_edges is not None and (k, i, j) in tabu_edges
                        else (
                            (0, 0)
                            if tabu_parent_nodes is not None and i in tabu_parent_nodes
                            else (
                                (0, 0)
                                if tabu_child_nodes is not None
                                and j in tabu_child_nodes
                                else (0, None)
                            )
                        )
                    )
                    for i in range(self.N)
                    for j in range(self.N)
                ]
            )
        bnds = bnds_w + bnds_a

        return bnds

    def get_all_parents(self):
        pass

    def graph_to_string_graph(self, graph: np.ndarray[float]) -> np.ndarray[str]:
        string_graph = np.array(["%.2f" % w for w in graph.reshape(graph.size)])
        string_graph = string_graph.reshape(graph.shape)

        string_graph[string_graph != "0.00"] = "-->"
        string_graph[string_graph == "0.00"] = ""

        return string_graph

    def _matrices_to_adjacency_matrix(
        self, w_est: np.ndarray, a_est: np.ndarray
    ) -> np.ndarray:
        """
        Converts the matrices output by dynotears (W and A) into a single adjacency matrix
        Args:
            w_est: Intra-slice weight matrix
            a_est: Inter-slice matrix

        Returns:
            Adjacency matrix representing the combined structure learnt
        """
        max_lag = w_est.shape[0] // a_est.shape[1]  # Determine max_lag
        n = a_est.shape[1]  # Number of nodes

        # Initialize adjacency matrix
        adjacency_matrix = np.zeros((n, n, max_lag + 1))

        # Populate intra-slice weights
        for i in range(w_est.shape[0]):
            lag = i % (max_lag + 1)
            for j in range(w_est.shape[1]):
                adjacency_matrix[j, i // (max_lag + 1), lag] = w_est[i, j]

        # Populate inter-slice weights
        for i in range(a_est.shape[0]):
            for j in range(a_est.shape[1]):
                adjacency_matrix[i + n, j, 0] = a_est[i, j]

        return adjacency_matrix

    def run_dynotears(
        self,
        lambda_w: float = 0.1,
        lambda_a: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.0,
        tau_max: int = 1,
        tabu_edges: List[Tuple[int, int, int]] = None,
        tabu_parent_nodes: List[int] = None,
        tabu_child_nodes: List[int] = None,
    ):
        self.tau_max = tau_max + 1
        X, Xlags = self._get_lags()
        bnds = self._get_bounds(
            tabu_edges, tabu_parent_nodes, tabu_child_nodes, X, Xlags
        )

        w_est, a_est = self._learn_dynamic_structure(
            X,
            Xlags,
            bnds,
            lambda_w,
            lambda_a,
            max_iter,
            h_tol,
        )
        a_est[np.abs(a_est) < w_threshold] = 0

        w_est[np.abs(w_est) < w_threshold] = 0

        a_est = a_est.reshape((self.tau_max - 1, self.N, self.N))
        a_est = np.moveaxis(a_est, 0, 2)

        a_est[np.abs(a_est) < w_threshold] = 0
        w_est[np.abs(w_est) < w_threshold] = 0
        graph = np.zeros((a_est.shape[1], a_est.shape[1], a_est.shape[2] + 1))
        graph[:, :, 1:] = a_est
        graph[:, :, 0] = w_est
        string_graph = self.graph_to_string_graph(graph)

        return w_est, a_est, string_graph, graph


def get_adj(links_coeffs, tau_max=None):
    if tau_max == None:
        tau_max = max(
            abs(value[i][0][1])
            for value in links_coeffs.values()
            for i in range(len(value))
        )
    n_vars = len(links_coeffs)
    adj = np.zeros((n_vars, n_vars, tau_max + 1))

    for key, value in links_coeffs.items():
        for i in range(len(value)):
            adj[value[i][0][0], key, abs(value[i][0][1])] = value[i][1]
    return adj


if __name__ == "__main__":
    import math
    from tigramite.toymodels import structural_causal_processes as toys

    np.random.seed(14)  # Fix random seed
    lin_f = lambda x: x
    links_coeffs = {
        0: [((0, -1), 0.7, lin_f)],
        1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, lin_f)],
        2: [((2, -1), 0.5, lin_f), ((0, -2), -0.5, lin_f)],
        3: [((3, -1), 0.0, lin_f)],
        4: [((4, -1), 0.0, lin_f), ((3, 0), 0.5, lin_f)],  # , ((3, -1), 0.3, lin_f)],
    }
    T = 200  # time series length
    # Make some noise with different variance, alternatively just noises=None
    noises = np.array(
        [
            (1.0 + 0.2 * float(j)) * np.random.randn((T + int(math.floor(0.2 * T))))
            for j in range(len(links_coeffs))
        ]
    ).T

    data, _ = toys.structural_causal_process(links_coeffs, T=T, noises=noises, seed=14)
    T, N = data.shape

    # For generality, we include some masking
    # mask = np.zeros(data.shape, dtype='int')
    # mask[:int(T/2)] = True
    mask = None

    # Initialize dataframe object, specify time axis and variable names
    var_names = [r"$X^0$", r"$X^1$", r"$X^2$", r"$X^3$", r"$X^4$"]
    dataframe = pp.DataFrame(
        data, mask=mask, datatime={0: np.arange(len(data))}, var_names=var_names
    )

    dynotears = Dynotears(dataframe=dataframe)
    w_est, a_est, graph, value_graph = dynotears.run_dynotears(
        tau_max=2, max_iter=100, w_threshold=0.2, lambda_a=0.1, lambda_w=0.1
    )
    tp.plot_graph(graph)
    fig, ax = tp.plot_time_series_graph(
        graph=graph,
        # graph = causal_effects.graph,
        link_width=None,
        cmap_edges="RdBu_r",
        arrow_linewidth=3,
        curved_radius=0.4,
        node_size=0.15,
        special_nodes=None,
        var_names=var_names,
        link_colorbar_label="Cross-strength",
        figsize=(4, 3),
    )

    a_true = get_adj(links_coeffs=links_coeffs, tau_max=2)

    tp.plot_time_series_graph(a_true)

    from causalnex.structure import dynotears as dyno

    import pandas as pd

    sm, w_est, a_est = dyno.from_numpy_dynamic(
        dynotears.X, dynotears.Xlags, w_threshold=0.1
    )

    print(sm.edges)
    # the structure model is in the form of a networkx graph it contains all the lags up to lag3 each lagged version of the variable is a node in the graph i want to create an adjacency matrix in th shape of (n_vars, n_vars, tau_max +1) where tau_max is the maximum lag

    adj = np.zeros((N, N, 4))

    for edge in sm.edges:
        # a node has the name 0_lag0, 3_lag2 etc
        # parse the node name to get the variable and the lag
        node1 = edge[0].split("_")
        node2 = edge[1].split("_")
        # get the variable and the lag
        var1 = int(node1[0])
        var2 = int(node2[0])
        lag1 = int(node1[1][3])
        lag2 = int(node2[1][3])
        # put into the adjacency matrix
        print(var1, var2, lag1, lag2)
        adj[var2, var1, lag1] = 1

    tp.plot_graph(adj)
    a_true = get_adj(links_coeffs=links_coeffs, tau_max=3)
    print(a_true[:, :, 2])
    tp.plot_graph(a_true)

    plt.show()
