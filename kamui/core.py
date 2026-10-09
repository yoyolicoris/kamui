"""Solvers for the phase unwrapping integer programs.

:func:`integrate` walks a directed graph accumulating edge weights,
:func:`calculate_k` and :func:`calculate_m` solve the two ILP formulations
as min-cost flows with LEMON where they can and with HiGHS otherwise, and
:func:`puma` runs the graph-cut PUMA algorithm when PyMaxflow is installed.
"""

import itertools
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


def _edge_weights(weights: np.ndarray, M: int) -> np.ndarray:
    """Return weights as an array, raising ValueError unless it has one entry per edge."""
    w = np.asarray(weights)
    if w.shape != (M,):
        raise ValueError(f"weights must have one entry per edge, shape ({M},); got {w.shape}")
    return w


def _lemon_rejects(w: np.ndarray) -> str | None:
    """Return what LEMON needs that the weights w lack, or None if it can take them exactly.

    Integer dtypes are exact as they are; floats must be finite whole
    numbers. The weights must also sum to less than 2**61, so that nothing
    LEMON adds up overflows int64: its potentials add path costs to an
    artificial cost of 2**62, and `calculate_m`'s supplies add up twice the
    weights of each vertex's edges.
    """
    if not np.issubdtype(w.dtype, np.integer) and not (
        np.isfinite(w).all() and np.array_equal(w, np.round(w))
    ):
        return "finite integer weights; scale and round them first, e.g. np.round(weights * 1000)"
    if np.min(w, initial=0) < 0:
        return "non-negative weights"
    if np.sum(w, dtype=np.float64) >= 2**61:
        return "weights that sum to less than 2**61"
    return None


def _orient_cycles(N: int, a: np.ndarray, b: np.ndarray, same: np.ndarray) -> np.ndarray | None:
    """Return which of N cycles to reverse so that shared edges are traversed both ways.

    Cycles ``a[i]`` and ``b[i]`` share an edge, which they traverse in the
    same direction where ``same[i]``. Exactly one of them must be reversed
    there, and neither or both elsewhere. Reversals are fixed along a
    spanning forest of the cycles, which `integrate` sums modulo 2 from a
    virtual root joined to one cycle per component. Returns None if no
    reversal works, as on a Möbius strip.
    """
    if not same.any():  # already consistent, as kamui's own grids are
        return np.zeros(N, dtype=bool)
    key, first = np.unique(np.minimum(a, b) * N + np.maximum(a, b), return_index=True)
    pairs = np.column_stack(np.divmod(key, N))
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
    """Return the dual-graph arc of each edge and the cycles to reverse, or None.

    Each cycle becomes a node, and node ``N`` stands for the outside of all
    cycles. Edge ``e`` runs from the cycle that traverses it forwards
    (``V[c, e] == 1``) to the one that traverses it backwards (``-1``), or to
    node ``N`` if only one cycle contains it; edges on no cycle get no arc
    (-1). That needs every edge on at most two cycles, as on 2-D grids and
    planar meshes, traversed in opposite directions once the cycles in the
    returned mask are reversed. Reversing cycle ``c`` negates row ``c`` of V
    and its residue, which leaves the program unchanged. Returns None when
    no such arcs exist.
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
    two = count[edge] == 2
    second = first[two] + 1
    r2 = np.full(edge.size, N, dtype=np.int64)
    r2[two] = V.indices[second]
    reverse = _orient_cycles(N, r1[two], r2[two], s1[two] == V.data[second])
    if reverse is None:
        return None
    forwards = (s1 > 0) != reverse[r1]
    tail = np.full(M, -1, dtype=np.int64)
    head = np.full(M, -1, dtype=np.int64)
    tail[edge] = np.where(forwards, r1, r2)
    head[edge] = np.where(forwards, r2, r1)
    return tail, head, reverse


def _sorted_graph(
    n_nodes: int, starts: np.ndarray, ends: np.ndarray
) -> tuple[pylmcf.Graph, np.ndarray]:
    """Return a pylmcf graph of the arcs, which it needs sorted by (start, end), and that order."""
    order = np.argsort(starts * n_nodes + ends, kind="stable").astype(np.int32)
    # int32 is LEMON's index type, so pylmcf takes these without a copy
    starts, ends = starts[order].astype(np.int32), ends[order].astype(np.int32)
    return pylmcf.Graph(n_nodes, edge_starts=starts, edge_ends=ends), order


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
    supply = np.append(b, -b.sum()).astype(np.int64)
    # pass the arcs as temporaries, so that only pylmcf's sorted copies outlive the call
    graph, order = _sorted_graph(
        supply.size,
        np.concatenate((tail[edge], head[edge])),
        np.concatenate((head[edge], tail[edge])),
    )
    graph.set_node_supply(supply)
    graph.set_edge_costs(np.tile(w[edge].astype(np.int64), 2)[order])
    # an optimal flow never carries more than the total supply on any arc
    capacity = np.abs(supply).sum() // 2 + 1
    graph.set_edge_capacities(np.full(order.size, capacity, dtype=np.int64))
    graph.solve()
    flow = np.empty(order.size, dtype=np.int64)
    flow[order] = graph.result()
    k[edge] = flow[: edge.size] - flow[edge.size :]
    return k


def _solve_offsets_by_flow(edges: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve ``min sum(w * |m[v] - m[u] + d|)`` with LEMON, reading m off as node potentials.

    The LP dual of this program is a circulation: a flow ``y[e]`` in
    ``[-w[e], w[e]]`` from u to v, conserved at every vertex, maximizing
    ``d @ y``. LEMON takes non-negative costs only, so with ``s = 1`` where
    ``d >= 0`` and ``-1`` elsewhere, each edge becomes one arc that carries
    ``g = w - s * y`` in ``[0, 2 w]`` at cost ``|d|``: from v to u when
    ``s = 1``, else from u to v, with the fixed flow ``s * w`` from u to v
    moved into the supplies. With ``m`` set to LEMON's potentials, that
    arc's reduced cost is ``s * (m[v] - m[u] + d)``, so complementary
    slackness makes the potentials an optimal m. A circulation always
    exists (``g = w``), so the flow is never infeasible. The weights must
    be non-negative integers.
    """
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    d, w = d.astype(np.int64), w.astype(np.int64)
    up = d >= 0
    starts, ends = np.where(up, v, u), np.where(up, u, v)
    fixed = np.where(up, w, -w)
    supply = np.zeros(int(edges.max()) + 1, dtype=np.int64)
    np.add.at(supply, v, fixed)
    np.add.at(supply, u, -fixed)
    graph, order = _sorted_graph(supply.size, starts, ends)
    graph.set_node_supply(supply)
    graph.set_edge_costs(np.abs(d)[order])
    graph.set_edge_capacities(2 * w[order])
    graph.solve()
    return graph.potentials()


def _vertex_indices(values: np.ndarray, name: str) -> np.ndarray:
    """Return values as int64, raising ValueError unless they are whole numbers."""
    if not np.issubdtype(values.dtype, np.integer) and not (
        np.isfinite(values).all() and np.array_equal(values, np.round(values))
    ):
        raise ValueError(f"{name} must hold integer vertex indices")
    return values.astype(np.int64, copy=False)


def _cycle_steps(simplices: Iterable[Iterable[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each step u -> v around the cycles, end to end, and the length of each cycle.

    Each cycle closes from its last vertex back to its first. An (S, k) array
    takes the vectorized path; other iterables are flattened first.
    """
    if isinstance(simplices, np.ndarray) and simplices.ndim == 2:
        cycles = _vertex_indices(simplices, "simplices")
        tails = np.roll(cycles, 1, axis=1).ravel()
        return tails, cycles.ravel(), np.full(len(cycles), cycles.shape[1])
    lengths = np.fromiter(map(len, simplices), dtype=np.int64, count=len(simplices))
    # read as floats, which hold every realistic vertex index exactly, so that
    # fractional ones are rejected rather than truncated
    vertices = itertools.chain.from_iterable(simplices)
    flat = np.fromiter(vertices, dtype=np.float64, count=int(lengths.sum()))
    heads = _vertex_indices(flat, "simplices")
    ends = np.cumsum(lengths)
    closed = lengths > 0
    previous = np.arange(-1, heads.size - 1)
    previous[(ends - lengths)[closed]] = ends[closed] - 1
    return heads[previous], heads, lengths


def _cycle_matrix(edges: np.ndarray, simplices: Iterable[Iterable[int]]) -> sp.csr_matrix:
    """Return the cycle-edge matrix V, raising ValueError for repeated or missing edges.

    ``V[c, e]`` is 1 where cycle ``c`` walks edge ``e = (u, v)`` from u to v,
    and -1 where it walks it from v to u. Edges are looked up by the code
    ``u * n + v`` in one sorted array, the forward direction first.
    """
    M = len(edges)
    edges = _vertex_indices(np.asarray(edges), "edges")
    tails, heads, lengths = _cycle_steps(simplices)
    n = max(int(np.max(edges, initial=-1)), int(np.max(heads, initial=-1))) + 1
    codes = edges[:, 0] * n + edges[:, 1]
    # Timsort merges presorted runs, such as a grid's horizontal and vertical
    # edges, in near-linear time, but is slower than quicksort on shuffled codes.
    presorted = np.count_nonzero(np.diff(codes) < 0) < 64
    order = np.argsort(codes, kind="stable" if presorted else "quicksort")
    sorted_codes = codes[order]
    if np.any(np.diff(sorted_codes) == 0):
        raise ValueError("edges must be unique")
    # a sentinel above every code, so that misses need no bounds check
    sorted_codes = np.append(sorted_codes, np.iinfo(np.int64).max)
    order = np.append(order, -1)

    def find(code: np.ndarray) -> np.ndarray:
        position = np.searchsorted(sorted_codes, code)
        missed = sorted_codes[position] != code
        # in place: the default mode="raise" would buffer the output anyway,
        # and every position is in range thanks to the sentinel
        np.take(order, position, out=position, mode="clip")
        position[missed] = -1
        return position

    cols = find(tails * n + heads)
    # a cycle's entries add up to at most its length, so int8 cannot wrap below 128
    vals = np.where(cols >= 0, 1, -1).astype(np.int8 if lengths.max(initial=0) < 128 else np.int64)
    backwards = np.flatnonzero(cols < 0)
    cols[backwards] = find(heads[backwards] * n + tails[backwards])
    if np.any(cols < 0):
        raise ValueError("simplices contain invalid edges")
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    V = sp.csr_matrix((vals, cols.astype(np.int32), indptr), shape=(lengths.size, M))
    V.sum_duplicates()  # a cycle that walks an edge twice adds up its entries
    return V


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
    *,
    solver: str = "auto",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, OptimizeResult] | None:
    """Solve per-edge integer ambiguities on elementary cycles.

    Finds per-edge integers ``k`` such that the corrected differences
    ``differences + k`` sum to zero around every simplex, minimizing the
    weighted L1 norm of ``k``.

    LEMON solves the program as a min-cost flow on the dual graph when
    every edge lies on at most two cycles. Otherwise HiGHS solves the LP
    relaxation first and keeps it when its optimum is integral. That holds
    for totally unimodular cycle matrices, such as those of 2-D grids and
    planar meshes, and for 3-D grids without a cyclical axis when
    ``differences`` are wrapped phase differences. Other cycle sets, such as
    3-D grids with a cyclical axis, can have fractional optima and are
    re-solved as an integer program.

    Parameters
    ----------
    edges : (M, 2) np.ndarray
        Array of edges; every edge must be unique.
    simplices : (N,) iterable of simplices
        Each element lists the vertices of one elementary cycle; every
        consecutive pair (closing the loop) must appear in ``edges`` in
        either direction. An (N, k) integer array of equal-length cycles
        is the fastest input.
    differences : (M,) np.ndarray
        Wrapped phase differences divided by the period; float or int.
    weights : (M,) np.ndarray, optional
        Per-edge weights. When None, each edge is weighted by its number of
        incident simplices with zero residue; pass ``np.ones(M)`` for
        uniform weights. Defaults to None.
    solver : {"auto", "highs", "lemon"}, optional
        "lemon" solves the program as a min-cost flow with LEMON's network
        simplex, through pylmcf. It needs every edge on at most two cycles,
        as on 2-D grids and planar meshes, with cycles that can be oriented
        to traverse each shared edge in opposite directions, and
        non-negative integer weights. The default weights are integers; to
        use fractional weights, scale and round them first. "highs" solves
        the program as a linear program with HiGHS. "auto" uses LEMON when
        the program allows it, and HiGHS otherwise. Defaults to "auto".
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
        ``edges``, ``weights`` does not have one entry per edge, ``solver``
        is unknown, or ``solver="lemon"`` and the
        weights are not non-negative integers that sum to less than 2**61,
        or the cycles do not form a min-cost flow.
    """
    _check_solver(solver)
    M = edges.shape[0]
    V = _cycle_matrix(edges, simplices)
    b_eq = -np.round(V @ differences).astype(np.int64)

    if weights is None:
        # each edge costs the number of its cycles that have no residue
        w = abs(V).T @ (np.abs(b_eq) == 0).astype(np.int64)
    else:
        w = _edge_weights(weights, M)

    arcs = None
    if solver != "highs":
        missing = _lemon_rejects(w)
        if missing is None:
            arcs = _dual_graph_arcs(V)
            if arcs is None:
                missing = (
                    "every edge on at most two cycles, and cycles that can be oriented to "
                    "traverse each shared edge in opposite directions"
                )
        if solver == "lemon" and missing:
            raise ValueError(f"solver='lemon' needs {missing}")
    if arcs is not None:
        del V  # LEMON needs only the arcs; free the matrix before it allocates
        tail, head, reverse = arcs
        try:
            k = _solve_min_cost_flow(tail, head, np.where(reverse, -b_eq, b_eq), w)
        except RuntimeError as err:
            k, info = None, OptimizeResult(fun=None, success=False, status=2, message=str(err))
        else:
            info = OptimizeResult(
                fun=float(np.abs(k) @ w.astype(np.float64)),
                success=True,
                status=0,
                message="Optimal min-cost flow found by LEMON's network simplex.",
            )
        info.update(ilp_fallback=False, solver="lemon")
        return (k, info) if return_info else k

    # k = x[:M] - x[M:] with x >= 0, so that the cost w @ |k| is linear
    A_eq = sp.hstack((V, -V), format="csr")
    x, info = _solve_integer_program(np.tile(w, 2), A_eq, b_eq)
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
        If ``weights`` does not have one entry per edge, ``solver`` is
        unknown, or ``solver="lemon"`` and the weights are not non-negative
        integers that sum to less than 2**61, or the absolute differences
        do not sum to less than 2**61.
    """
    _check_solver(solver)
    if not np.issubdtype(differences.dtype, np.integer):
        raise TypeError(f"differences must have an integer dtype; got {differences.dtype}")
    M = edges.shape[0]
    N = np.max(edges) + 1
    w = np.ones((M,), dtype=np.int64) if weights is None else _edge_weights(weights, M)
    if solver != "highs":
        missing = _lemon_rejects(w)
        # the absolute differences are LEMON's costs, bounded like the weights
        if missing is None and np.abs(differences.astype(np.float64)).sum() >= 2**61:
            missing = "differences whose absolute values sum to less than 2**61"
        if solver == "lemon" and missing:
            raise ValueError(f"solver='lemon' needs {missing}")
        if missing is None:
            m = _solve_offsets_by_flow(edges, differences, w)
            info = OptimizeResult(
                fun=float(
                    np.abs(m[edges[:, 1]] - m[edges[:, 0]] + differences) @ w.astype(np.float64)
                ),
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
    c = np.concatenate((np.zeros(N, dtype=np.int64), w, w))

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
