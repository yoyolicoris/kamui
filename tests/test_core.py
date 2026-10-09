from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import Delaunay

import kamui.core as core
from kamui import wrap_difference
from kamui.core import calculate_k, calculate_m, integrate, puma
from kamui.utils import get_2d_edges_and_simplices, get_3d_edges_and_simplices


def _failed_solve(x, status=2, message="The problem is infeasible."):
    # stand-in for a linprog result that HiGHS did not report as optimal
    return SimpleNamespace(x=x, success=False, fun=None, status=status, message=message)


def test_integrate_chain():
    # integrate is called with edges in both directions (as in
    # unwrap_arbitrary), so every vertex appears as a source
    edges = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
    weights = np.array([1.0, 2.0, 4.0, -1.0, -2.0, -4.0])
    np.testing.assert_allclose(integrate(edges, weights), [0.0, 1.0, 3.0, 7.0])


def test_integrate_sums_along_tree_paths():
    # The tree branches at the root, so the visit order steps between the two
    # branches, from one node to another that shares no edge with it. Each
    # node must be accumulated from its own parent.
    edges = np.array([[0, 1], [1, 2], [0, 3], [3, 4]])
    weights = np.array([1.0, 2.0, 4.0, 8.0])
    both = np.concatenate((edges, np.flip(edges, 1)))
    result = integrate(both, np.concatenate((weights, -weights)))
    np.testing.assert_allclose(result, [0.0, 1.0, 3.0, 4.0, 12.0])


def test_integrate_leaves_unreachable_nodes_at_zero():
    # the only edge points into the start node, so nothing else is reached
    np.testing.assert_array_equal(integrate(np.array([[1, 0]]), np.array([5.0])), [0.0, 0.0])


def _triangle():
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    simplices = [[0, 1, 2]]
    return edges, simplices


def test_calculate_k_recovers_loop_residue():
    edges, simplices = _triangle()
    differences = np.array([0.4, 0.4, 0.3])  # loop sum 1.1, so the residue is 1
    k = calculate_k(edges, simplices, differences, weights=np.ones(3))
    assert k is not None
    assert k.sum() == -1
    assert np.abs(k).sum() == 1


def test_calculate_k_weight_options_agree_on_trivial_loop():
    edges, simplices = _triangle()
    differences = np.array([0.1, 0.1, -0.2])  # loop sum 0, so k is 0
    k_default = calculate_k(edges, simplices, differences)
    k_weighted = calculate_k(edges, simplices, differences, weights=np.ones(3))
    for k in (k_default, k_weighted):
        np.testing.assert_array_equal(k, [0, 0, 0])


def test_calculate_k_accepts_reversed_simplex_edges():
    edges, _ = _triangle()
    k = calculate_k(edges, [[0, 2, 1]], np.array([0.1, 0.1, -0.2]))
    np.testing.assert_array_equal(k, [0, 0, 0])


def test_calculate_k_rejects_duplicate_edges():
    with pytest.raises(ValueError, match="edges must be unique"):
        calculate_k(np.array([[0, 1], [0, 1]]), [[0, 1]], np.array([0.1, 0.2]))


def test_calculate_k_rejects_invalid_simplex_edge():
    with pytest.raises(ValueError, match="simplices contain invalid edges"):
        calculate_k(np.array([[0, 1]]), [[0, 2]], np.array([0.1]))


def _spy_linprog(monkeypatch):
    # record the integrality argument of every solve
    calls = []
    real = core.linprog

    def spy(*args, **kwargs):
        calls.append(kwargs.get("integrality"))
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "linprog", spy)
    return calls


def test_calculate_k_solves_grid_as_plain_lp(monkeypatch):
    # A 2-D grid's cycle matrix is totally unimodular, so one LP solve
    # suffices. This is a speed canary: if a future HiGHS returns a
    # non-vertex optimum here, results stay correct through the ILP fallback
    # and only this assertion fails.
    calls = _spy_linprog(monkeypatch)
    edges, simplices = get_2d_edges_and_simplices((6, 6))
    psi = np.random.default_rng(0).uniform(-np.pi, np.pi, 36)
    differences = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]]) / (2 * np.pi)
    k = calculate_k(edges, simplices, differences, solver="highs")
    assert calls == [None]
    assert np.abs(k).sum() > 0


def test_calculate_k_falls_back_to_ilp_on_fractional_lp(monkeypatch):
    # Two 4-cycles of K4 whose cycle matrix is not totally unimodular. The
    # unique LP optimum puts -0.5 on the cheap edges (0, 1) and (2, 3);
    # rounding it would leave the first loop's residue uncorrected.
    calls = _spy_linprog(monkeypatch)
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    psi = np.array([0.0, 0.49, -0.49, 0.0])
    differences = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], period=1.0)
    weights = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 1.0])
    k = calculate_k(edges, [[0, 1, 2, 3], [0, 1, 3, 2]], differences, weights=weights)
    assert calls == [None, 1]
    c = differences + k
    np.testing.assert_allclose([c[0] + c[3] + c[5] - c[2], c[0] + c[4] - c[5] - c[1]], 0, atol=1e-9)
    # the integer optimum moves one expensive edge
    assert weights @ np.abs(k) == 10


def test_calculate_k_reports_the_ilp_fallback():
    # the K4 program above: the LP optimum is fractional, so the report must
    # say the integer program was solved, at its optimal cost of 10
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    psi = np.array([0.0, 0.49, -0.49, 0.0])
    differences = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], period=1.0)
    weights = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 1.0])
    simplices = [[0, 1, 2, 3], [0, 1, 3, 2]]
    k, info = calculate_k(edges, simplices, differences, weights=weights, return_info=True)
    assert info.success and info.ilp_fallback
    assert info.fun == pytest.approx(10.0)
    assert info.fun == pytest.approx(weights @ np.abs(k))


def test_calculate_k_returns_none_when_infeasible(monkeypatch):
    edges, simplices = _triangle()
    monkeypatch.setattr(core, "linprog", lambda *a, **k: _failed_solve(x=None))
    assert calculate_k(edges, simplices, np.array([0.1, 0.1, -0.2]), solver="highs") is None


def _loop_sums(edges, simplices, values):
    # sum of the oriented edge values around each simplex
    index = {tuple(e): i for i, e in enumerate(edges.tolist())}
    sums = []
    for simplex in simplices:
        simplex = list(simplex)
        total = 0.0
        for u, v in zip(simplex[-1:] + simplex[:-1], simplex):
            total += values[index[(u, v)]] if (u, v) in index else -values[index[(v, u)]]
        sums.append(total)
    return np.array(sums)


def _wrapped_differences(edges, n_vertices, seed):
    psi = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_vertices)
    return wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]]) / (2 * np.pi)


def _delaunay_mesh(seed=0, n=150):
    points = np.random.default_rng(seed).uniform(0, 1, (n, 2))
    triangles = Delaunay(points).simplices
    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return np.unique(np.sort(pairs, axis=1), axis=0), triangles.tolist(), n


def _planar(case):
    if case == "Delaunay mesh":
        return _delaunay_mesh()
    cyclical_axis = {"grid, cyclical axis 0": 0, "grid, both axes cyclical": (0, 1)}.get(case, ())
    return *get_2d_edges_and_simplices((9, 8), cyclical_axis=cyclical_axis), 72


@pytest.mark.parametrize(
    "case",
    [
        "grid",
        "grid, cyclical axis 0",
        "grid, both axes cyclical",
        "Delaunay mesh",
        "integer weights",
    ],
)
def test_calculate_k_lemon_matches_highs(case):
    # LEMON solves the planar programs as min-cost flows; HiGHS must reach the
    # same optimal cost, and LEMON's k must close every loop.
    edges, simplices, n_vertices = _planar(case)
    differences = _wrapped_differences(edges, n_vertices, seed=1)
    weights = None
    if case == "integer weights":
        weights = np.random.default_rng(2).integers(1, 10, len(edges))
    k, info = calculate_k(edges, simplices, differences, weights=weights, return_info=True)
    _, reference = calculate_k(
        edges, simplices, differences, weights=weights, solver="highs", return_info=True
    )
    assert info.solver == "lemon" and reference.solver == "highs"
    assert info.success and not info.ilp_fallback
    assert reference.fun > 0
    assert info.fun == pytest.approx(reference.fun)
    np.testing.assert_allclose(_loop_sums(edges, simplices, differences + k), 0, atol=1e-9)


def test_calculate_k_uses_highs_when_cycles_are_not_a_network():
    # an interior edge of a 3-D grid lies on four cycles
    edges, simplices = get_3d_edges_and_simplices((3, 3, 3))
    differences = _wrapped_differences(edges, 27, seed=3)
    k, info = calculate_k(edges, simplices, differences, return_info=True)
    assert info.solver == "highs"
    np.testing.assert_allclose(_loop_sums(edges, simplices, differences + k), 0, atol=1e-9)
    with pytest.raises(ValueError, match="at most two"):
        calculate_k(edges, simplices, differences, solver="lemon")


def _reverse_some(simplices, seed):
    flip = np.random.default_rng(seed).random(len(simplices)) < 0.5
    return [list(s)[::-1] if f else list(s) for s, f in zip(simplices, flip)]


@pytest.mark.parametrize("case", ["grid", "grid, both axes cyclical", "Delaunay mesh"])
def test_calculate_k_lemon_reorients_cycles(case):
    # Cycles listed in either direction, as other meshing tools may give them,
    # are reversed until shared edges run both ways, so LEMON still applies.
    edges, simplices, n_vertices = _planar(case)
    simplices = _reverse_some(simplices, seed=6)
    differences = _wrapped_differences(edges, n_vertices, seed=7)
    # uniform weights: random phases leave residues nearly everywhere, which
    # can zero every default weight and make any k optimal
    uniform = np.ones(len(edges))
    k, info = calculate_k(edges, simplices, differences, weights=uniform, return_info=True)
    _, reference = calculate_k(
        edges, simplices, differences, weights=uniform, solver="highs", return_info=True
    )
    assert info.solver == "lemon" and info.success
    assert reference.fun > 0
    assert info.fun == pytest.approx(reference.fun)
    np.testing.assert_allclose(_loop_sums(edges, simplices, differences + k), 0, atol=1e-9)


def _mobius_strip(n=4):
    # A strip of n squares whose ends are glued with a half twist: top vertex
    # i is i, bottom vertex i is n + i, and the top of square n - 1 continues
    # into the bottom of square 0. No orientation of the squares makes every
    # shared edge run both ways.
    top = list(range(n)) + [n]
    bottom = list(range(n, 2 * n)) + [0]
    squares = [[top[i], top[i + 1], bottom[i + 1], bottom[i]] for i in range(n)]
    pairs = {tuple(sorted((s[j - 1], s[j]))) for s in squares for j in range(4)}
    return np.array(sorted(pairs)), squares


def test_calculate_k_uses_highs_on_a_mobius_strip():
    edges, squares = _mobius_strip()
    differences = _wrapped_differences(edges, 8, seed=8)
    k, info = calculate_k(edges, squares, differences, return_info=True)
    assert info.solver == "highs"
    np.testing.assert_allclose(_loop_sums(edges, squares, differences + k), 0, atol=1e-9)
    with pytest.raises(ValueError, match="oriented"):
        calculate_k(edges, squares, differences, solver="lemon")


def test_calculate_k_uses_highs_for_a_cycle_that_repeats_an_edge():
    # traversing the triangle twice puts a 2 in the cycle matrix
    edges, _ = _triangle()
    _, info = calculate_k(edges, [[0, 1, 2, 0, 1, 2]], np.array([0.4, 0.4, 0.3]), return_info=True)
    assert info.solver == "highs"


def test_calculate_k_leaves_negative_weights_to_highs():
    edges, simplices = _triangle()
    differences = np.array([0.4, 0.4, 0.3])
    k, info = calculate_k(edges, simplices, differences, weights=-np.ones(3), return_info=True)
    assert info.solver == "highs"
    assert k is None  # unbounded
    with pytest.raises(ValueError, match="non-negative"):
        calculate_k(edges, simplices, differences, weights=-np.ones(3), solver="lemon")


def test_calculate_k_without_cycles_keeps_k_at_zero():
    k, info = calculate_k(np.array([[0, 1], [1, 2]]), [], np.array([0.4, -0.3]), return_info=True)
    assert info.solver == "lemon"
    np.testing.assert_array_equal(k, [0, 0])


def test_calculate_k_reports_an_infeasible_flow():
    # The four faces of a tetrahedron close up, so no edge reaches the outside.
    # These (not wrapped) differences round to face residues that do not sum
    # to zero, so no k satisfies the constraints.
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    face_matrix = np.zeros((4, 6))
    index = {tuple(e): i for i, e in enumerate(edges.tolist())}
    for row, face in enumerate(faces):
        for u, v in zip(face[-1:] + face[:-1], face):
            face_matrix[row, index.get((u, v), index.get((v, u)))] += 1 if (u, v) in index else -1
    differences = np.linalg.lstsq(face_matrix, np.array([0.6, -0.3, -0.3, 0.0]), rcond=None)[0]
    k, info = calculate_k(edges, faces, differences, return_info=True)
    assert k is None
    assert info.solver == "lemon" and not info.success and "INFEASIBLE" in info.message
    assert calculate_k(edges, faces, differences, solver="highs") is None


def test_calculate_k_leaves_fractional_weights_to_highs():
    # LEMON needs integer costs, and kamui does not round weights itself: with
    # solver="auto" fractional weights go to HiGHS, with "lemon" they raise
    edges, simplices = get_2d_edges_and_simplices((6, 6))
    differences = _wrapped_differences(edges, 36, seed=4)
    weights = np.random.default_rng(5).uniform(0.1, 1.0, len(edges))
    k, info = calculate_k(edges, simplices, differences, weights=weights, return_info=True)
    assert info.solver == "highs"
    np.testing.assert_allclose(_loop_sums(edges, simplices, differences + k), 0, atol=1e-9)
    with pytest.raises(ValueError, match="integer weights"):
        calculate_k(edges, simplices, differences, weights=weights, solver="lemon")
    # whole numbers stored as floats are integer weights
    _, info = calculate_k(
        edges, simplices, differences, weights=np.round(weights * 10), return_info=True
    )
    assert info.solver == "lemon"


def test_calculate_k_rejects_unknown_solver():
    edges, simplices = _triangle()
    with pytest.raises(ValueError, match="solver must be"):
        calculate_k(edges, simplices, np.array([0.4, 0.4, 0.3]), solver="cplex")


def test_calculate_m_chain():
    edges = np.array([[0, 1], [1, 2]])
    differences = np.array([1, -1], dtype=np.int64)
    m = calculate_m(edges, differences)
    assert m is not None
    assert m.dtype == np.int64
    np.testing.assert_array_equal(m[edges[:, 0]] - m[edges[:, 1]], differences)


def test_calculate_m_accepts_weights():
    edges = np.array([[0, 1], [1, 2]])
    differences = np.array([1, -1], dtype=np.int64)
    m = calculate_m(edges, differences, weights=np.array([2.0, 3.0]))
    np.testing.assert_array_equal(m[edges[:, 0]] - m[edges[:, 1]], differences)


def test_calculate_m_accepts_any_integer_dtype():
    edges = np.array([[0, 1], [1, 2]])
    differences = np.array([1, -1], dtype=np.int32)
    m = calculate_m(edges, differences)
    np.testing.assert_array_equal(m[edges[:, 0]] - m[edges[:, 1]], differences)


def test_calculate_m_rejects_non_integer_differences():
    with pytest.raises(TypeError, match="integer dtype"):
        calculate_m(np.array([[0, 1]]), np.array([0.5]))


def test_calculate_m_reports_lp_solution():
    edges = np.array([[0, 1], [1, 2]])
    m, info = calculate_m(
        edges, np.array([1, -1], dtype=np.int64), solver="highs", return_info=True
    )
    np.testing.assert_array_equal(m[edges[:, 0]] - m[edges[:, 1]], [1, -1])
    assert info.success and not info.ilp_fallback and info.solver == "highs"
    assert info.fun == pytest.approx(0.0)


def test_calculate_m_reports_failure(monkeypatch):
    monkeypatch.setattr(core, "linprog", lambda *a, **k: _failed_solve(x=None))
    m, info = calculate_m(
        np.array([[0, 1]]), np.array([1], dtype=np.int64), solver="highs", return_info=True
    )
    assert m is None
    assert not info.success and info.status == 2 and "infeasible" in info.message


def test_calculate_m_rejects_non_optimal_solution(monkeypatch):
    # e.g. a time limit: HiGHS returns a point but does not report success
    result = _failed_solve(x=np.zeros(4), status=1, message="Time limit reached.")
    monkeypatch.setattr(core, "linprog", lambda *a, **k: result)
    assert calculate_m(np.array([[0, 1]]), np.array([1], dtype=np.int64), solver="highs") is None


def test_calculate_m_returns_none_when_infeasible(monkeypatch):
    monkeypatch.setattr(core, "linprog", lambda *a, **k: _failed_solve(x=None))
    assert calculate_m(np.array([[0, 1]]), np.array([1], dtype=np.int64), solver="highs") is None


@pytest.mark.parametrize(
    "case", ["grid", "integer weights", "zero weights", "Delaunay mesh", "two components"]
)
def test_calculate_m_lemon_matches_highs(case):
    # LEMON solves the dual circulation and returns its node potentials as m;
    # they must reach HiGHS's optimal cost.
    rng = np.random.default_rng(9)
    if case == "Delaunay mesh":
        edges, _, n_vertices = _delaunay_mesh()
    else:
        edges, _ = get_2d_edges_and_simplices((9, 8))
        n_vertices = 72
    if case == "two components":
        edges = np.concatenate((edges, edges + n_vertices))
        n_vertices *= 2
    psi = rng.uniform(-np.pi, np.pi, n_vertices)
    differences = np.round((psi[edges[:, 1]] - psi[edges[:, 0]]) / (2 * np.pi)).astype(np.int64)
    weights = np.ones(len(edges), dtype=np.int64)
    if case == "integer weights":
        weights = rng.integers(1, 10, len(edges))
    elif case == "zero weights":
        weights = rng.integers(0, 3, len(edges))
    m, info = calculate_m(edges, differences, weights, return_info=True)
    _, reference = calculate_m(edges, differences, weights, solver="highs", return_info=True)
    assert info.solver == "lemon" and reference.solver == "highs"
    assert info.success and not info.ilp_fallback
    assert m.dtype == np.int64 and m.shape == (n_vertices,)
    assert reference.fun > 0
    assert info.fun == pytest.approx(reference.fun)


def test_calculate_m_leaves_fractional_and_negative_weights_to_highs():
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    differences = np.array([1, 0, 0], dtype=np.int64)
    m, info = calculate_m(edges, differences, np.array([0.5, 1.0, 1.0]), return_info=True)
    assert info.solver == "highs"
    assert info.fun == pytest.approx(0.5)
    with pytest.raises(ValueError, match="integer weights"):
        calculate_m(edges, differences, np.array([0.5, 1.0, 1.0]), solver="lemon")
    m, info = calculate_m(edges, differences, -np.ones(3), return_info=True)
    assert info.solver == "highs" and m is None  # unbounded
    with pytest.raises(ValueError, match="non-negative"):
        calculate_m(edges, differences, -np.ones(3), solver="lemon")


def test_calculate_m_rejects_unknown_solver():
    with pytest.raises(ValueError, match="solver must be"):
        calculate_m(np.array([[0, 1]]), np.array([1], dtype=np.int64), solver="cplex")


def test_lemon_keeps_large_integer_weights_exact():
    # 2**53 + 1 rounds to 2**53 in float64, which would tie the two cheapest
    # edges; the cheapest edge is 1 only if the weights stay exact
    edges, simplices = _triangle()
    weights = np.array([2**53 + 1, 2**53, 2**53 + 2])
    k, info = calculate_k(edges, simplices, np.array([0.4, 0.4, 0.3]), weights, return_info=True)
    assert info.solver == "lemon"
    np.testing.assert_array_equal(k, [0, -1, 0])
    differences = np.array([1, 1, 1])
    m, info = calculate_m(edges, differences, weights, return_info=True)
    assert info.solver == "lemon"
    np.testing.assert_array_equal(m[edges[:, 1]] - m[edges[:, 0]] + differences, [0, 3, 0])


@pytest.mark.parametrize(
    ("weight", "message"), [(np.inf, "finite integer"), (2.0**61, "sum to less than 2\\*\\*61")]
)
def test_lemon_rejects_weights_it_cannot_take_exactly(weight, message):
    edges, simplices = _triangle()
    weights = np.array([1.0, weight, 1.0])
    with pytest.raises(ValueError, match=message):
        calculate_k(edges, simplices, np.array([0.4, 0.4, 0.3]), weights, solver="lemon")
    with pytest.raises(ValueError, match=message):
        calculate_m(edges, np.array([1, 0, 0]), weights, solver="lemon")


def test_lemon_leaves_weights_that_overflow_its_sums_to_highs():
    # Each weight is below pylmcf's 2**62 cost limit, but on an 8x8 grid LEMON
    # reported calculate_k's feasible flow as infeasible, and calculate_m's
    # vertex supplies overflowed int64. Such weights now go to HiGHS, which
    # reports for itself whether it can solve them.
    edges, simplices = get_2d_edges_and_simplices((8, 8))
    differences = _wrapped_differences(edges, 64, seed=10)
    weights = np.full(len(edges), 2**61)
    _, info = calculate_k(edges, simplices, differences, weights, return_info=True)
    assert info.solver == "highs"
    _, info = calculate_m(
        edges, np.round(differences * 3).astype(np.int64), weights, return_info=True
    )
    assert info.solver == "highs"


def test_puma_leaves_consistent_phase_alone():
    psi = np.array([0.0, 0.5])
    edges = np.array([[0, 1]])
    np.testing.assert_array_equal(puma(psi, edges), [0.0, 0.0])


def test_puma_accepts_larger_max_jump():
    psi = np.array([0.0, 0.5, 1.0])
    edges = np.array([[0, 1], [1, 2]])
    k = puma(psi, edges, max_jump=2)
    assert k.shape == (3,)


def test_puma_accepts_energy_decrease():
    psi = np.array([0.0, -0.4, 0.3, -0.2])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    k = puma(psi, edges, max_jump=1)
    np.testing.assert_array_equal(k, [1.0, 1.0, 0.0, 1.0])
    i, j = edges[:, 0], edges[:, 1]
    before = np.sum(np.abs(psi[j] - psi[i]))
    after = np.sum(np.abs(k[j] - k[i] - psi[i] + psi[j]))
    assert after < before


def test_puma_reports_final_energy():
    psi = np.array([0.0, -0.4, 0.3, -0.2])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    k, info = puma(psi, edges, return_info=True)
    i, j = edges[:, 0], edges[:, 1]
    assert info.fun == pytest.approx(np.sum(np.abs(k[j] - k[i] - psi[i] + psi[j])))
    assert info.success and info.nit >= 2


def test_puma_requires_pymaxflow(monkeypatch):
    monkeypatch.setattr(core, "maxflow", None)
    with pytest.raises(ImportError, match="PyMaxflow"):
        puma(np.array([0.0, 0.5]), np.array([[0, 1]]))
