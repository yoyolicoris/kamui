"""Solvers for the phase unwrapping integer programs.

:func:`integrate` walks a directed graph accumulating edge weights,
:func:`calculate_k` and :func:`calculate_m` solve the two ILP formulations
with HiGHS, and :func:`puma` runs the graph-cut PUMA algorithm when
PyMaxflow is installed.
"""

from collections.abc import Iterable

import numpy as np
import pylmcf
import scipy.sparse as sp
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csgraph as csg

try:
    import maxflow
except ImportError:  # pragma: no cover (PyMaxflow is always installed in the test env)
    maxflow = None

__all__ = ["integrate", "calculate_k", "calculate_m", "puma"]


def _solve_integer_program(
    c: np.ndarray, A_eq: sp.csr_matrix, b_eq: np.ndarray
) -> tuple[np.ndarray | None, OptimizeResult]:
    """Minimize ``c @ x`` subject to ``A_eq @ x == b_eq``, ``x >= 0``, x integer.

    Try the LP relaxation first: it is several times cheaper than branch and
    bound, and an integral LP optimum is also optimal for the integer
    program. Not every input gives one (3-D grids with a cyclical axis and
    arbitrary user cycles can have half-integral optima), so re-solve with
    integrality constraints whenever HiGHS returns a fractional solution.
    The solution is None unless HiGHS reports an optimal one: on a time or
    iteration limit it can hand back a point that is not optimal. Also
    returns the solver report that ``return_info`` exposes.
    """
    res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    ilp_fallback = bool(res.success and np.abs(res.x - np.round(res.x)).max() > 1e-6)
    if ilp_fallback:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    info = OptimizeResult(
        fun=res.fun,
        success=res.success,
        status=res.status,
        message=res.message,
        ilp_fallback=ilp_fallback,
        solver="highs",
    )
    return (np.round(res.x) if res.success else None), info


def _check_solver(solver: str) -> None:
    """Raise ValueError unless solver names a backend."""
    if solver not in ("auto", "highs", "lemon"):
        raise ValueError(f"solver must be 'auto', 'highs' or 'lemon'; got {solver!r}")


def _lemon_takes(w: np.ndarray, solver: str) -> bool:
    """Return whether LEMON can take the weights w; with solver="lemon", raise if not."""
    integer = np.array_equal(w, np.round(w))  # also False for NaN
    non_negative = np.min(w, initial=0) >= 0
    if solver == "lemon" and not integer:
        raise ValueError(
            "solver='lemon' needs integer weights; scale and round them first, "
            "e.g. np.round(weights * 1000)"
        )
    if solver == "lemon" and not non_negative:
        raise ValueError("solver='lemon' needs non-negative weights")
    return solver != "highs" and integer and non_negative


def _orient_cycles(N: int, a: np.ndarray, b: np.ndarray, same: np.ndarray) -> np.ndarray | None:
    """Return which of N cycles to reverse so that shared edges are traversed both ways.

    Cycles ``a[i]`` and ``b[i]`` share an edge, which they traverse in the
    same direction where ``same[i]``. Exactly one of them must be reversed
    there, and neither or both elsewhere. Reversals are fixed along a
    spanning forest of the cycles, which `integrate` sums modulo 2 from a
    virtual root joined to one cycle per component. Returns None if no
    reversal works, as on a Möbius strip.
    """
    if a.size == 0:
        return np.zeros(N, dtype=bool)
    pairs = np.column_stack((np.minimum(a, b), np.maximum(a, b)))
    pairs, first = np.unique(pairs, axis=0, return_index=True)
    parity = same[first].astype(np.int64)
    adjacency = sp.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
    _, label = csg.connected_components(adjacency, directed=False)
    _, roots = np.unique(label, return_index=True)
    links = np.column_stack((np.full_like(roots, N), roots))
    flips = integrate(
        np.concatenate((pairs, pairs[:, ::-1], links)),
        np.concatenate((parity, parity, np.zeros(roots.size, dtype=np.int64))),
        start_i=N,
    )
    reverse = flips[:N] % 2 == 1
    if np.any((reverse[a] != reverse[b]) != same):
        return None
    return reverse


def _dual_graph_arcs(V: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the dual-graph arc of each edge, or None if V is not a network matrix.

    Each cycle becomes a node, and node ``N`` stands for the outside of all
    cycles. Edge ``e`` runs from the cycle that traverses it forwards
    (``V[c, e] == 1``) to the one that traverses it backwards (``-1``), or to
    node ``N`` if only one cycle contains it. That needs every edge on at most
    two cycles, in opposite directions, as on 2-D grids and planar meshes.
    Edges on no cycle get no arc (-1).

    Cycles that traverse a shared edge the same way are first reversed where
    `_orient_cycles` can make every shared edge run both ways; the returned
    mask marks them. Reversing cycle ``c`` negates row ``c`` of V and its
    residue, which leaves the program unchanged.
    """
    N, M = V.shape
    V = V.tocsc()
    V.eliminate_zeros()
    count = np.diff(V.indptr)
    if count.max(initial=0) > 2 or np.any(np.abs(V.data) != 1):
        return None
    edge = np.flatnonzero(count > 0)
    first = V.indptr[edge]
    r1, s1 = V.indices[first].astype(np.int64), V.data[first]
    r2, s2 = np.full(edge.size, N, dtype=np.int64), -s1
    two = count[edge] == 2
    r2[two], s2[two] = V.indices[first[two] + 1], V.data[first[two] + 1]
    reverse = _orient_cycles(N, r1[two], r2[two], s1[two] == s2[two])
    if reverse is None:
        return None
    forwards = (s1 > 0) != reverse[r1]
    tail = np.full(M, -1, dtype=np.int64)
    head = np.full(M, -1, dtype=np.int64)
    tail[edge] = np.where(forwards, r1, r2)
    head[edge] = np.where(forwards, r2, r1)
    return tail, head, reverse


def _solve_min_cost_flow(
    tail: np.ndarray, head: np.ndarray, b: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Solve ``min w @ |k|`` subject to ``V k == b`` as a min-cost flow with LEMON.

    ``tail`` and ``head`` are the dual-graph arcs from `_dual_graph_arcs`, and
    ``k[e]`` is the flow from tail to head minus the flow back. Edges without
    an arc keep ``k = 0``. The weights must be non-negative integers. Raises
    RuntimeError if the flow is infeasible.
    """
    k = np.zeros(tail.size, dtype=np.int64)
    edge = np.flatnonzero(tail >= 0)
    if edge.size == 0:
        return k
    cost = w[edge].astype(np.int64)
    starts = np.concatenate((tail[edge], head[edge]))
    ends = np.concatenate((head[edge], tail[edge]))
    order = np.lexsort((ends, starts))  # pylmcf wants arcs sorted by (start, end)
    supply = np.append(b, -b.sum()).astype(np.int64)
    graph = pylmcf.Graph(len(supply), edge_starts=starts[order], edge_ends=ends[order])
    graph.set_node_supply(supply)
    graph.set_edge_costs(np.tile(cost, 2)[order])
    # an optimal flow never carries more than the total supply on any arc
    capacity = np.abs(supply).sum() // 2 + 1
    graph.set_edge_capacities(np.full(starts.size, capacity, dtype=np.int64))
    graph.solve()
    flow = np.empty(starts.size, dtype=np.int64)
    flow[order] = graph.result()
    k[edge] = flow[: edge.size] - flow[edge.size :]
    return k


def _solve_offsets_by_flow(edges: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve ``min sum(w * |m[v] - m[u] + d|)`` with LEMON, reading m off as node potentials.

    The LP dual of this program is a circulation: a flow ``y[e]`` in
    ``[-w[e], w[e]]`` from u to v, conserved at every vertex, maximizing
    ``d @ y``. LEMON takes non-negative costs only, so each edge becomes one
    arc that carries ``g = w - sign(d) * y`` in ``[0, 2 w]`` at cost ``|d|``:
    from v to u when ``d >= 0``, else from u to v, with the fixed flow
    ``sign(d) * w`` from u to v moved into the supplies. The reduced cost of
    that arc is then ``|m[v] - m[u] + d|`` with the right sign, so
    complementary slackness makes LEMON's potentials an optimal m. A
    circulation always exists (``g = w``), so the flow is never infeasible.
    The weights must be non-negative integers.
    """
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    d, w = d.astype(np.int64), w.astype(np.int64)
    up = d >= 0
    starts, ends = np.where(up, v, u), np.where(up, u, v)
    fixed = np.where(up, w, -w)
    supply = np.zeros(int(edges.max()) + 1, dtype=np.int64)
    np.add.at(supply, v, fixed)
    np.add.at(supply, u, -fixed)
    order = np.lexsort((ends, starts))  # pylmcf wants arcs sorted by (start, end)
    graph = pylmcf.Graph(supply.size, edge_starts=starts[order], edge_ends=ends[order])
    graph.set_node_supply(supply)
    graph.set_edge_costs(np.abs(d)[order])
    graph.set_edge_capacities(2 * w[order])
    graph.solve()
    return graph.potentials()


def integrate(edges: np.ndarray, weights: np.ndarray, start_i: int = 0) -> np.ndarray:
    """Integrate edge weights along the breadth-first spanning tree of a directed graph.

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
        Sum of the edge weights on the tree path from ``start_i`` to each of
        the V nodes; unreachable nodes keep 0.
    """
    N = int(edges.max()) + 1
    G = sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(N, N))
    # Any spanning tree gives the same sums when the weights are consistent
    # around every cycle; a breadth-first tree is shallow, which keeps the
    # pointer jumping below to few passes.
    order, parent = csg.breadth_first_order(G, start_i, directed=True, return_predecessors=True)

    # Start from the weight of the tree edge into each reached node. Each node
    # must build on its parent, not on the node visited just before it: two
    # consecutive nodes in visit order often share no edge.
    result = np.zeros(N, dtype=weights.dtype)
    children = order[1:]
    if children.size:
        result[children] = np.asarray(G[parent[children], children]).ravel()

    # Pointer jumping: each pass adds the partial sum held by a node's current
    # ancestor and then skips to that ancestor's ancestor, so a tree of depth
    # d is summed in about log2(d) vectorized passes.
    ancestor = parent.copy()
    pending = np.flatnonzero(ancestor >= 0)
    while pending.size:
        up = ancestor[pending]
        result[pending] += result[up]
        ancestor[pending] = ancestor[up]
        pending = pending[ancestor[pending] >= 0]
    return result


def calculate_k(
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]],
    differences: np.ndarray,
    weights: np.ndarray | None = None,
    adaptive_weighting: bool = True,
    *,
    solver: str = "auto",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, OptimizeResult] | None:
    """Solve per-edge integer ambiguities on elementary cycles.

    Finds per-edge integers ``k`` such that the corrected differences
    ``differences + k`` sum to zero around every simplex, minimizing the
    weighted L1 norm of ``k`` through HiGHS.

    The LP relaxation is solved first and kept when its optimum is
    integral. That holds for 2-D grids (cyclical axes included) and planar
    meshes, whose cycle matrices are totally unimodular, and for 3-D grids
    without a cyclical axis when ``differences`` are wrapped phase
    differences. Other cycle sets, such as 3-D grids with a cyclical axis,
    can have fractional optima and are re-solved as an integer program.

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
    solver : {"auto", "highs", "lemon"}, optional
        "lemon" solves the program as a min-cost flow with LEMON's network
        simplex, through pylmcf. It needs every edge on at most two cycles,
        traversed in opposite directions, as on 2-D grids and planar
        meshes, and non-negative integer weights. The default adaptive and
        uniform weights are integers; to use fractional weights, scale and
        round them first. "highs" solves the program as a linear program
        with HiGHS. "auto" uses LEMON when the program allows it, and
        HiGHS otherwise. Defaults to "auto".
    return_info : bool, optional
        Also return the solver report. Defaults to False.

    Returns
    -------
    k : (M,) np.ndarray or None
        Integer ambiguity per edge, or None if the solver finds no optimal
        solution, e.g. because the program is infeasible.
    info : scipy.optimize.OptimizeResult
        Only returned if ``return_info`` is True. ``fun`` is the weighted
        L1 cost of ``k`` (None without a solution); ``success``,
        ``status`` and ``message`` come from the solver, which ``solver``
        names ("lemon" or "highs"); ``ilp_fallback`` is True when the LP
        optimum was fractional and HiGHS solved the integer program
        instead.

    Raises
    ------
    ValueError
        If the edges are not unique, the simplices use an edge not in
        ``edges``, ``solver`` is unknown, or ``solver="lemon"`` and the
        weights are not non-negative integers or the cycles do not form
        a min-cost flow.
    """
    _check_solver(solver)
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

    w = np.asarray(c[:M], dtype=np.float64)
    arcs = _dual_graph_arcs(V) if _lemon_takes(w, solver) else None
    if solver == "lemon" and arcs is None:
        raise ValueError(
            "solver='lemon' needs every edge on at most two cycles, and cycles that can "
            "be oriented to traverse each shared edge in opposite directions"
        )
    if arcs is not None:
        tail, head, reverse = arcs
        try:
            k = _solve_min_cost_flow(tail, head, np.where(reverse, -b_eq, b_eq), w)
        except RuntimeError as err:
            k, info = None, OptimizeResult(fun=None, success=False, status=2, message=str(err))
        else:
            info = OptimizeResult(
                fun=float(w @ np.abs(k)),
                success=True,
                status=0,
                message="Optimal min-cost flow found by LEMON's network simplex.",
            )
        info.update(ilp_fallback=False, solver="lemon")
        return (k, info) if return_info else k

    x, info = _solve_integer_program(c, A_eq, b_eq)
    k = None if x is None else (x[:M] - x[M:]).astype(np.int64)
    return (k, info) if return_info else k


def calculate_m(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    solver: str = "auto",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, OptimizeResult] | None:
    """Solve per-vertex integer offsets from quantized edge differences.

    Finds per-vertex integers ``m`` with ``m[u] - m[v]`` matching
    ``differences`` on each edge ``(u, v)``, minimizing the weighted L1
    norm of the slacks.

    LEMON solves the LP dual of this program, a min-cost circulation, and
    ``m`` is read off its node potentials. HiGHS solves the program itself:
    its constraint matrix (an edge-node incidence matrix beside two
    identity blocks) is totally unimodular, so the LP relaxation already
    has an integral optimum; the integer program is only a fallback.

    Parameters
    ----------
    edges : (M, 2) np.ndarray
        Array of edges.
    differences : (M,) np.ndarray
        Quantized differences, of any integer dtype.
    weights : (M,) np.ndarray, optional
        Per-edge weights. Defaults to uniform weights.
    solver : {"auto", "highs", "lemon"}, optional
        "lemon" solves the dual circulation with LEMON's network simplex,
        through pylmcf, and needs non-negative integer weights; to use
        fractional weights, scale and round them first. "highs" solves the
        program as a linear program with HiGHS. "auto" uses LEMON when the
        weights allow it, and HiGHS otherwise. Defaults to "auto".
    return_info : bool, optional
        Also return the solver report. Defaults to False.

    Returns
    -------
    m : (V,) np.ndarray or None
        Integer offset per vertex, or None if HiGHS finds no optimal
        solution, e.g. because the program is infeasible.
    info : scipy.optimize.OptimizeResult
        Only returned if ``return_info`` is True. ``fun`` is the weighted
        L1 norm of the slacks (None without a solution); ``success``,
        ``status`` and ``message`` come from the solver, which ``solver``
        names ("lemon" or "highs"); ``ilp_fallback`` is True when the LP
        optimum was fractional and HiGHS solved the integer program
        instead.

    Raises
    ------
    TypeError
        If ``differences`` does not have an integer dtype.
    ValueError
        If ``solver`` is unknown, or ``solver="lemon"`` and the weights are
        not non-negative integers.
    """
    _check_solver(solver)
    if not np.issubdtype(differences.dtype, np.integer):
        raise TypeError(f"differences must have an integer dtype; got {differences.dtype}")
    M = edges.shape[0]
    N = np.max(edges) + 1
    if weights is None:
        weights = np.ones((M,), dtype=np.int64)

    w = np.asarray(weights, dtype=np.float64)
    if _lemon_takes(w, solver):
        m = _solve_offsets_by_flow(edges, differences, w)
        info = OptimizeResult(
            fun=float(w @ np.abs(m[edges[:, 1]] - m[edges[:, 0]] + differences)),
            success=True,
            status=0,
            message="Optimal node potentials found by LEMON's network simplex.",
            ilp_fallback=False,
            solver="lemon",
        )
        return (m, info) if return_info else m

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
    c = np.concatenate((np.zeros(N, dtype=np.int64), weights, weights))

    x, info = _solve_integer_program(c, A_eq, b_eq=differences)
    m = None if x is None else x[:N].astype(np.int64)
    return (m, info) if return_info else m


def puma(
    psi: np.ndarray,
    edges: np.ndarray,
    max_jump: int = 1,
    p: float = 1,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, OptimizeResult]:
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
    return_info : bool, optional
        Also return a report on the minimization. Defaults to False.

    Returns
    -------
    K : (N,) np.ndarray
        Integer jump per vertex.
    info : scipy.optimize.OptimizeResult
        Only returned if ``return_info`` is True. ``fun`` is the final
        energy and ``nit`` the number of max-flow solves.

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
    nit = 0

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
            nit += 1

            partition = G.get_grid_segments(np.arange(total_nodes))
            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break
    if not return_info:
        return K
    info = OptimizeResult(
        fun=prev_Ek,
        success=True,
        status=0,
        message="No proposed jump lowers the energy further.",
        nit=nit,
    )
    return K, info
