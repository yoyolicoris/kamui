"""Solvers for the phase unwrapping integer programs.

:func:`integrate` walks a directed graph accumulating edge weights,
:func:`calculate_k` and :func:`calculate_m` solve the two ILP formulations
with HiGHS, and :func:`puma` runs the graph-cut PUMA algorithm when
PyMaxflow is installed.
"""

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse import csgraph as csg

try:
    import maxflow
except ImportError:  # pragma: no cover (PyMaxflow is always installed in the test env)
    maxflow = None

__all__ = ["integrate", "calculate_k", "calculate_m", "puma"]


def integrate(edges: np.ndarray, weights: np.ndarray, start_i: int = 0) -> np.ndarray:
    """Integrate edge weights along a depth-first traversal of a directed graph.

    Parameters
    ----------
    edges : (N, 2) np.ndarray
        Directed edges as (source, target) index pairs.
    weights : (N,) np.ndarray
        Weight of each edge.
    start_i : int, optional
        Index of the node the traversal starts from. Defaults to 0.

    Returns
    -------
    (V,) np.ndarray
        Accumulated weight at each of the V nodes; unreachable nodes keep 0.
    """
    G = sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])))
    N = max(G.shape)
    result = np.zeros(N, dtype=weights.dtype)

    nodes = csg.depth_first_order(G, start_i, directed=True, return_predecessors=False)

    pairs = np.stack([nodes[:-1], nodes[1:]], axis=1)
    for u, v in pairs:
        result[v] = result[u] + G[u, v]
    return result


def calculate_k(
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]],
    differences: np.ndarray,
    weights: np.ndarray | None = None,
    adaptive_weighting: bool = True,
) -> np.ndarray | None:
    """Solve per-edge integer ambiguities on elementary cycles.

    Finds per-edge integers ``k`` such that the corrected differences
    ``differences + k`` sum to zero around every simplex, minimizing the
    weighted L1 norm of ``k`` through HiGHS.

    The constraint matrix is totally unimodular, so the linear
    programming relaxation already attains the integer optimum and no
    integer constraints are imposed.

    Parameters
    ----------
    edges : (M, 2) np.ndarray
        Array of edges; every edge must be unique.
    simplices : (N,) iterable of simplices
        Each element lists the vertices of one elementary cycle; every
        consecutive pair (closing the loop) must appear in ``edges`` in
        either direction.
    differences : (M,) np.ndarray
        Wrapped phase differences divided by the period; float or int.
    weights : (M,) np.ndarray, optional
        Per-edge weights. When None, each edge is weighted by its number of
        incident simplices with zero residue if ``adaptive_weighting`` is
        set, else uniformly. Defaults to None.
    adaptive_weighting : bool, optional
        Weight edges by incident zero-residue simplex counts.
        Defaults to True.

    Returns
    -------
    (M,) np.ndarray or None
        Integer ambiguity per edge, or None if HiGHS reports infeasibility.
    """
    M, N = edges.shape[0], len(simplices)

    edge_dict = {tuple(x): i for i, x in enumerate(edges)}
    if len(edge_dict) != M:
        raise ValueError("edges must be unique")
    rows = []
    cols = []
    vals = []
    for i, simplex in enumerate(simplices):
        u = simplex[-1]
        for v in simplex:
            key = (u, v)
            rows.append(i)
            if key in edge_dict:
                cols.append(edge_dict[key])
                vals.append(1)
            else:
                try:
                    cols.append(edge_dict[(v, u)])
                except KeyError:
                    raise ValueError("simplices contain invalid edges")
                vals.append(-1)
            u = v
    rows = np.array(rows)
    cols = np.array(cols)
    vals = np.array(vals)

    V = sp.csr_matrix((vals, (rows, cols)), shape=(N, M))
    y = V @ differences
    y = np.round(y).astype(np.int64)

    A_eq = sp.csr_matrix(
        (
            np.concatenate((vals, -vals)),
            (np.tile(rows, 2), np.concatenate((cols, cols + M))),
        ),
        shape=(N, M * 2),
    )
    b_eq = -y

    if weights is None:
        if adaptive_weighting:
            nonzero_simplices = np.minimum(np.abs(b_eq), 1)
            W = np.abs(A_eq)
            num_nonzero_simplices = nonzero_simplices @ W
            num_simplices = W.sum(0).A1
            c = num_simplices - num_nonzero_simplices
        else:
            c = np.ones((M * 2,), dtype=np.int64)
    else:
        c = np.tile(weights, 2)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    if res.x is None:
        return None
    k = res.x[:M] - res.x[M:]
    k = np.round(k).astype(np.int64)
    return k


def calculate_m(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve per-vertex integer offsets from quantized edge differences.

    Finds per-vertex integers ``m`` with ``m[u] - m[v]`` matching
    ``differences`` on each edge ``(u, v)``, minimizing the weighted L1
    norm of the slacks through HiGHS.

    The constraint matrix is totally unimodular, so the linear
    programming relaxation already attains the integer optimum and no
    integer constraints are imposed.

    Parameters
    ----------
    edges : (M, 2) np.ndarray
        Array of edges.
    differences : (M,) np.ndarray
        Quantized differences; must have int64 dtype.
    weights : (M,) np.ndarray, optional
        Per-edge weights. Defaults to uniform weights.

    Returns
    -------
    (V,) np.ndarray or None
        Integer offset per vertex, or None if HiGHS reports infeasibility.
    """
    assert differences.dtype == np.int64, "differences must be int"
    M = edges.shape[0]
    N = np.max(edges) + 1

    vals = np.concatenate((np.ones((M,), dtype=np.int64), -np.ones((M,), dtype=np.int64)))
    rows = np.tile(np.arange(M), 2)
    cols = np.concatenate((edges[:, 0], edges[:, 1]))

    A_eq = sp.csr_matrix(
        (
            np.concatenate((vals, np.ones(M), -np.ones(M))).astype(np.int64),
            (
                np.tile(rows, 2),
                np.concatenate((cols, np.arange(2 * M) + N)),
            ),
        ),
        shape=(M, N + 2 * M),
    )
    if weights is None:
        weights = np.ones((M,), dtype=np.int64)
    c = np.concatenate((np.zeros(N, dtype=np.int64), weights, weights))

    b_eq = differences

    res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    if res.x is None:
        return None
    m = res.x[:N]
    return np.round(m).astype(np.int64)


def puma(psi: np.ndarray, edges: np.ndarray, max_jump: int = 1, p: float = 1) -> np.ndarray:
    """Unwrap phase with PUMA, a graph-cut based phase unwrapping method.

    Iteratively proposes integer jumps on subsets of vertices and keeps
    them when a max-flow/min-cut solve lowers the ``p``-norm energy.

    Parameters
    ----------
    psi : (N,) np.ndarray
        Wrapped phase in units of the period.
    edges : (M, 2) np.ndarray
        Array of edges.
    max_jump : int, optional
        Maximum jump step proposed per iteration. Defaults to 1.
    p : float, optional
        Norm order of the energy. Defaults to 1.

    Returns
    -------
    (N,) np.ndarray
        Integer jump per vertex.

    Raises
    ------
    ImportError
        If PyMaxflow is not installed.
    """
    if maxflow is None:
        raise ImportError("puma requires PyMaxflow; install with `pip install kamui[extra]`")
    if max_jump > 1:
        jump_steps = list(range(1, max_jump + 1)) * 2
    else:
        jump_steps = [max_jump]
    total_nodes = psi.size

    def V(x):
        """Absolute value raised to the p-th power.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            ``np.abs(x) ** p``.
        """
        return np.abs(x) ** p

    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        """Total p-norm energy of the current integer jumps.

        Parameters
        ----------
        K : (N,) np.ndarray
            Integer jumps per vertex.
        psi : (N,) np.ndarray
            Wrapped phase in units of the period.
        i : (M,) np.ndarray
            Source vertex of each edge.
        j : (M,) np.ndarray
            Target vertex of each edge.

        Returns
        -------
        float
            Sum of ``|K[j] - K[i] - psi[i] + psi[j]| ** p`` over edges.
        """
        return np.sum(V(K[j] - K[i] - psi[i] + psi[j]))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    for step in jump_steps:
        while True:
            G = maxflow.Graph[float]()
            G.add_nodes(total_nodes)

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - step)
            e10 = V(a + step)
            weight = np.maximum(0, e10 + e01 - e00 - e11)

            G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

            a = e10 - e00
            flip_mask = a < 0
            tmp_st_weight = np.zeros((2, total_nodes))
            flip_index = np.stack((flip_mask.astype(int), 1 - flip_mask.astype(int)), axis=1)
            positive_a = np.where(flip_mask, -a, a)
            np.add.at(tmp_st_weight, (flip_index.ravel(), edges.ravel()), positive_a.repeat(2))

            for i in range(total_nodes):
                G.add_tedge(i, tmp_st_weight[0, i], tmp_st_weight[1, i])
            G.maxflow()

            partition = G.get_grid_segments(np.arange(total_nodes))
            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break
    return K
