import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

import kamui
from kamui import (
    get_2d_edges_and_simplices,
    unwrap_arbitrary,
    unwrap_dimensional,
    wrap_difference,
)
from kamui.utils import _merge_weights


def test_version_wired():
    assert kamui.__version__
    assert "__version__" in kamui.__all__


def test_wrap_difference_bounds():
    rng = np.random.default_rng(0)
    x = rng.uniform(-20, 20, size=1000)
    w = wrap_difference(x)
    assert np.all(w >= -np.pi)
    assert np.all(w < np.pi)


def test_wrap_difference_endpoints():
    np.testing.assert_allclose(wrap_difference(np.array([-np.pi, np.pi])), [-np.pi, -np.pi])


def test_wrap_difference_period_invariance():
    rng = np.random.default_rng(1)
    x = rng.uniform(-20, 20, size=100)
    k = rng.integers(-3, 4, size=100)
    np.testing.assert_allclose(wrap_difference(x + k * 2 * np.pi), wrap_difference(x))


def test_wrap_difference_custom_period():
    x = np.array([0.0, 0.6, 1.4, 2.0])
    np.testing.assert_allclose(wrap_difference(x, period=2.0), [0.0, 0.6, -0.6, 0.0])


def _ramp_2d(n=6, m=6):
    yy, xx = np.mgrid[0:n, 0:m]
    return 1.5 * xx + 1.2 * yy


def test_unwrap_dimensional_2d():
    true_phase = _ramp_2d()
    result = unwrap_dimensional(wrap_difference(true_phase))
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_edgelist():
    true_phase = _ramp_2d(5, 5)
    result = unwrap_dimensional(wrap_difference(true_phase), use_edgelist=True)
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_weights_and_start_pixel():
    true_phase = _ramp_2d()
    weights = np.linspace(0.5, 1.0, true_phase.size).reshape(true_phase.shape)
    result = unwrap_dimensional(wrap_difference(true_phase), start_pixel=(2, 3), weights=weights)
    assert result is not None
    np.testing.assert_allclose(result - result[2, 3], true_phase - true_phase[2, 3], atol=1e-6)


@pytest.mark.parametrize("use_edgelist", [False, True])
def test_unwrap_dimensional_integer_weights_reach_lemon(use_edgelist):
    rng = np.random.default_rng(4)
    true_phase = _ramp_2d(12, 12) + rng.normal(0, 1.3, (12, 12))
    coherence = rng.uniform(0.2, 1.0, true_phase.shape)
    wrapped = wrap_difference(true_phase)
    # fractional weights go to HiGHS, and raise with solver="lemon"
    _, info = unwrap_dimensional(
        wrapped, weights=coherence, use_edgelist=use_edgelist, return_info=True
    )
    assert info.solver == "highs"
    with pytest.raises(ValueError, match="integer weights"):
        unwrap_dimensional(wrapped, weights=coherence, use_edgelist=use_edgelist, solver="lemon")
    # integer pixel weights stay integer per edge, so LEMON takes them
    weights = np.round(coherence * 100)
    result, info = unwrap_dimensional(
        wrapped, weights=weights, use_edgelist=use_edgelist, return_info=True
    )
    _, reference = unwrap_dimensional(
        wrapped, weights=weights, use_edgelist=use_edgelist, solver="highs", return_info=True
    )
    assert info.solver == "lemon" and info.success
    assert info.fun == pytest.approx(reference.fun)
    np.testing.assert_allclose(wrap_difference(result - wrapped), 0, atol=1e-9)


def test_unwrap_dimensional_cyclical():
    n, m = 6, 8
    ii, jj = np.mgrid[0:n, 0:m]
    true_phase = 2.5 * np.sin(2 * np.pi * ii / n) + 0.4 * jj
    result = unwrap_dimensional(wrap_difference(true_phase), cyclical_axis=0)
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_3d():
    n = 3
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    true_phase = 1.2 * (xx + yy + zz)
    result = unwrap_dimensional(wrap_difference(true_phase))
    assert result is not None
    np.testing.assert_allclose(
        result - result[0, 0, 0], true_phase - true_phase[0, 0, 0], atol=1e-6
    )


def test_unwrap_dimensional_gc():
    true_phase = _ramp_2d()
    result = unwrap_dimensional(wrap_difference(true_phase), method="gc")
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_gc_weights(monkeypatch):
    true_phase = _ramp_2d()
    weights = np.linspace(0.1, 1.0, true_phase.size).reshape(true_phase.shape)
    captured = {}
    orig_puma = kamui.puma

    def mock_puma(*args, **kwargs):
        captured.update(kwargs)
        return orig_puma(*args, **kwargs)

    monkeypatch.setattr(kamui, "puma", mock_puma)
    result, info = unwrap_dimensional(
        wrap_difference(true_phase), method="gc", weights=weights, return_info=True
    )
    assert result is not None
    assert info.success and info.nit >= 1
    assert "weights" in captured
    edges, _ = get_2d_edges_and_simplices(true_phase.shape)
    expected = _merge_weights(weights, edges, "sum")
    np.testing.assert_array_equal(captured["weights"], expected)
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_3d_ramp_on_larger_grid():
    # 10x10x10 is large enough for consecutive voxels in the traversal order
    # not to be adjacent, which broke integrate before #22
    zz, yy, xx = np.mgrid[0:10, 0:10, 0:10]
    true_phase = 1.3 * zz + 0.9 * yy - 1.1 * xx
    result = unwrap_dimensional(wrap_difference(true_phase))
    np.testing.assert_allclose(
        result - result[0, 0, 0], true_phase - true_phase[0, 0, 0], atol=1e-6
    )


def test_unwrap_dimensional_rejects_mismatched_start_pixel():
    with pytest.raises(ValueError, match="one index per dimension"):
        unwrap_dimensional(np.zeros((4, 4)), start_pixel=(0, 0, 0))


def test_unwrap_dimensional_rejects_1d():
    with pytest.raises(ValueError, match="2D or 3D"):
        unwrap_dimensional(np.zeros(5))


def test_unwrap_dimensional_propagates_none(monkeypatch):
    monkeypatch.setattr(kamui, "unwrap_arbitrary", lambda *a, **k: None)
    assert unwrap_dimensional(np.zeros((4, 4))) is None


def test_unwrap_arbitrary_ilp_with_simplices():
    psi = np.array([0.0, 0.5, 1.0])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    result = unwrap_arbitrary(psi, edges, [[0, 1, 2]])
    np.testing.assert_allclose(result - result[0], psi - psi[0], atol=1e-9)


def test_unwrap_arbitrary_ilp_edgelist():
    true_phase = np.array([0.0, 2.0, 4.0, 6.0])
    psi = wrap_difference(true_phase)
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    result = unwrap_arbitrary(psi, edges, None)
    np.testing.assert_allclose(result, true_phase, atol=1e-9)


def test_unwrap_arbitrary_ilp_edgelist_infeasible(monkeypatch):
    monkeypatch.setattr(kamui, "calculate_m", lambda *a, **k: (None, OptimizeResult(success=False)))
    result = unwrap_arbitrary(np.zeros(3), np.array([[0, 1], [1, 2]]), None)
    assert result is None


def test_unwrap_arbitrary_ilp_simplex_infeasible(monkeypatch):
    monkeypatch.setattr(kamui, "calculate_k", lambda *a, **k: (None, OptimizeResult(success=False)))
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    result = unwrap_arbitrary(np.zeros(3), edges, [[0, 1, 2]])
    assert result is None


def _jittered_point_cloud(rng, n=20, spacing=0.15):
    yy, xx = np.mgrid[0:n, 0:n] * spacing
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return points + rng.uniform(-0.3, 0.3, points.shape) * spacing


def _delaunay_graph(points, max_edge=0.3):
    # Drop the long sliver triangles Delaunay adds along the convex hull, so
    # the true phase changes by well under pi along every remaining edge.
    simplices = Delaunay(points).simplices
    pairs = np.stack([simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [1, 2]]], axis=1)
    lengths = np.linalg.norm(points[pairs[..., 0]] - points[pairs[..., 1]], axis=-1)
    keep = lengths.max(axis=1) <= max_edge
    edges = np.unique(np.sort(pairs[keep].reshape(-1, 2), axis=1), axis=0)
    return edges, simplices[keep].tolist()


def test_unwrap_arbitrary_point_cloud_is_exact_in_any_order():
    # issue #9: on a Delaunay point cloud, the simplex path gave wrong results
    # that changed with the input order of the points
    rng = np.random.default_rng(0)
    points = _jittered_point_cloud(rng)
    for order in (np.arange(len(points)), rng.permutation(len(points))):
        p = points[order]
        true_phase = 3 * p[:, 0] - 2 * p[:, 1]
        psi = wrap_difference(true_phase)
        edges, simplices = _delaunay_graph(p)
        result = unwrap_arbitrary(psi, edges, simplices)
        np.testing.assert_allclose(result - result[0], true_phase - true_phase[0], atol=1e-6)


def test_unwrap_arbitrary_reports_matching_costs_on_both_ilp_paths():
    # With uniform weights, the simplex and edgelist programs minimize the
    # same cost over the same corrections, so their optima must agree
    # (the comparison asked for in #9).
    rng = np.random.default_rng(3)
    true_phase = _ramp_2d(12, 12) + rng.normal(0, 1.3, (12, 12))
    psi = wrap_difference(true_phase).ravel()
    edges, simplices = get_2d_edges_and_simplices((12, 12))
    with_cycles, info_k = unwrap_arbitrary(
        psi, edges, simplices, weights=np.ones(len(edges)), return_info=True
    )
    edgelist, info_m = unwrap_arbitrary(psi, edges, None, return_info=True)
    assert info_k.success and info_m.success
    assert info_k.solver == info_m.solver == "lemon"
    assert info_k.fun > 0
    assert info_k.fun == pytest.approx(info_m.fun)
    np.testing.assert_allclose(wrap_difference(with_cycles - psi), 0, atol=1e-9)
    np.testing.assert_allclose(wrap_difference(edgelist - psi), 0, atol=1e-9)


def test_unwrap_arbitrary_reports_failure(monkeypatch):
    monkeypatch.setattr(kamui, "calculate_k", lambda *a, **k: (None, OptimizeResult(success=False)))
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    result, info = unwrap_arbitrary(np.zeros(3), edges, [[0, 1, 2]], return_info=True)
    assert result is None and not info.success


def test_unwrap_arbitrary_gc_reports_energy():
    result, info = unwrap_arbitrary(
        np.array([0.0, 1.0]), np.array([[0, 1]]), method="gc", return_info=True
    )
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-9)
    assert info.success and info.nit >= 1


def test_unwrap_dimensional_reports_info():
    true_phase = _ramp_2d()
    result, info = unwrap_dimensional(wrap_difference(true_phase), return_info=True)
    assert result.shape == true_phase.shape
    assert info.success and not info.ilp_fallback
    np.testing.assert_allclose(result - result[0, 0], true_phase - true_phase[0, 0], atol=1e-6)


def test_unwrap_dimensional_reports_failure(monkeypatch):
    failed = OptimizeResult(success=False)
    monkeypatch.setattr(kamui, "unwrap_arbitrary", lambda *a, **k: (None, failed))
    assert unwrap_dimensional(np.zeros((4, 4)), return_info=True) == (None, failed)


def test_unwrap_arbitrary_gc():
    psi = np.array([0.0, 1.0])
    edges = np.array([[0, 1]])
    result = unwrap_arbitrary(psi, edges, method="gc")
    np.testing.assert_allclose(result, psi, atol=1e-9)


def test_unwrap_arbitrary_gc_with_weights():
    psi = np.array([0.0, 1.0])
    edges = np.array([[0, 1]])
    weights = np.array([2.5])
    result, info = unwrap_arbitrary(psi, edges, method="gc", weights=weights, return_info=True)
    np.testing.assert_allclose(result, psi, atol=1e-9)
    expected_fun = float(weights[0] * (np.abs(psi[1] - psi[0]) / (2 * np.pi)))
    assert info.success and info.fun == pytest.approx(expected_fun)


@pytest.mark.parametrize("extra", [[1, 0], [0, 1]])
def test_unwrap_arbitrary_rejects_repeated_edges_before_solving(monkeypatch, extra):
    # dev summed both copies of such an edge in integrate and returned a wrong
    # phase; now the simplex path refuses them before calling the solver
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], extra])
    monkeypatch.setattr(kamui, "calculate_k", lambda *a, **k: pytest.fail("solver ran"))
    with pytest.raises(ValueError, match="each edge once"):
        unwrap_arbitrary(np.zeros(4), edges, [[0, 1, 2, 3]])


def test_unwrap_arbitrary_rejects_bad_method():
    with pytest.raises(ValueError, match="method must be"):
        unwrap_arbitrary(np.zeros(2), np.array([[0, 1]]), method="bad")


@pytest.mark.parametrize("method", ["ilp", "gc"])
def test_unwrap_arbitrary_disconnected_components(method):
    true_phase = np.r_[np.arange(5) * 2.0, 50 + np.arange(5) * 2.0, 99.0]
    psi = wrap_difference(true_phase)
    # 0..4 connected, 5..9 connected, node 10 is isolated
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9]])
    weights = np.ones(len(edges), dtype=np.float64 if method == "gc" else np.int64)

    # simplex path (for ILP) and default GC
    simplices = [] if method == "ilp" else None
    res, info = unwrap_arbitrary(
        psi, edges, simplices=simplices, method=method, weights=weights, start_i=0, return_info=True
    )
    assert res is not None
    assert info.success
    np.testing.assert_allclose(np.diff(res[:5]), 2.0, atol=1e-9)
    np.testing.assert_allclose(np.diff(res[5:10]), 2.0, atol=1e-9)
    np.testing.assert_allclose(res[0], psi[0], atol=1e-9)
    np.testing.assert_allclose(res[5], psi[5], atol=1e-9)
    np.testing.assert_allclose(res[10], psi[10], atol=1e-9)

    # edgelist path (for ILP)
    if method == "ilp":
        res_m, info_m = unwrap_arbitrary(
            psi, edges, simplices=None, method="ilp", start_i=0, return_info=True
        )
        assert res_m is not None
        assert info_m.success and info_m.solver == "lemon"
        np.testing.assert_allclose(np.diff(res_m[:5]), 2.0, atol=1e-9)
        np.testing.assert_allclose(np.diff(res_m[5:10]), 2.0, atol=1e-9)
        np.testing.assert_allclose(res_m[0], psi[0], atol=1e-9)
        np.testing.assert_allclose(res_m[5], psi[5], atol=1e-9)
        np.testing.assert_allclose(res_m[10], psi[10], atol=1e-9)

    # start_i in the second component (start_i=7)
    res_start7 = unwrap_arbitrary(psi, edges, simplices=simplices, method=method, start_i=7)
    assert res_start7 is not None
    np.testing.assert_allclose(np.diff(res_start7[:5]), 2.0, atol=1e-9)
    np.testing.assert_allclose(np.diff(res_start7[5:10]), 2.0, atol=1e-9)
    np.testing.assert_allclose(res_start7[0], psi[0], atol=1e-9)
    np.testing.assert_allclose(res_start7[7], psi[7], atol=1e-9)
    np.testing.assert_allclose(res_start7[10], psi[10], atol=1e-9)


def test_unwrap_arbitrary_disconnected_simplices_2d_and_list():
    true_phase = np.array([0.0, 1.0, 0.5, 5.0, 6.0, 5.5, 99.0])
    psi = wrap_difference(true_phase)
    # nodes 0..2 form cycle 1, nodes 3..5 form cycle 2, node 6 is isolated
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
    simplices_2d = np.array([[0, 1, 2], [3, 4, 5]])
    simplices_list = [[0, 1, 2], [3, 4, 5]]

    res_2d, info_2d = unwrap_arbitrary(psi, edges, simplices=simplices_2d, return_info=True)
    assert res_2d is not None and info_2d.success
    np.testing.assert_allclose(res_2d[:3] - res_2d[0], true_phase[:3] - true_phase[0], atol=1e-9)
    np.testing.assert_allclose(res_2d[3:6] - res_2d[3], true_phase[3:6] - true_phase[3], atol=1e-9)
    np.testing.assert_allclose(res_2d[6], psi[6], atol=1e-9)

    res_list, info_list = unwrap_arbitrary(psi, edges, simplices=simplices_list, return_info=True)
    assert res_list is not None and info_list.success
    np.testing.assert_allclose(res_list, res_2d, atol=1e-9)


def test_unwrap_arbitrary_start_i_isolated():
    psi = np.array([0.0, 1.0, 5.0])
    edges = np.array([[0, 1]])
    res = unwrap_arbitrary(psi, edges, start_i=2)
    np.testing.assert_allclose(res, psi, atol=1e-9)


def test_unwrap_arbitrary_disconnected_ilp_fallback():
    c1_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    c1_psi = np.array([0.0, 0.49, -0.49, 0.0]) * (2 * np.pi)
    c1_weights = np.array([1.0, 10.0, 10.0, 10.0, 10.0, 1.0])
    c1_simplices = [[0, 1, 2, 3], [0, 1, 3, 2]]

    c2_edges = np.array([[4, 5]])
    c2_psi = np.array([0.0, 1.0])
    c2_weights = np.array([2.0])

    edges = np.vstack([c1_edges, c2_edges])
    psi = np.concatenate([c1_psi, c2_psi])
    weights = np.concatenate([c1_weights, c2_weights])
    simplices = c1_simplices

    res, info = unwrap_arbitrary(psi, edges, simplices=simplices, weights=weights, return_info=True)
    assert res is not None
    assert info.success
    assert info.ilp_fallback
    assert "highs" in info.solver


def test_unwrap_arbitrary_isolated_vertices():
    psi = np.array([0.5, 1.2, -0.8])
    empty_edges = np.empty((0, 2), dtype=np.int64)
    res = unwrap_arbitrary(psi, empty_edges)
    np.testing.assert_allclose(res, psi, atol=1e-9)

    # also with method="gc"
    res_gc = unwrap_arbitrary(psi, empty_edges, method="gc")
    np.testing.assert_allclose(res_gc, psi, atol=1e-9)


@pytest.mark.parametrize("use_edgelist", [False, True])
@pytest.mark.parametrize("method", ["ilp", "gc"])
def test_unwrap_dimensional_nan_grid_2d(use_edgelist, method):
    n = 8
    g = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    true_phase = 1.2 * g[0] + 0.8 * g[1]
    wrapped = wrap_difference(true_phase)
    wrapped[:, 3:5] = np.nan

    unwrapped = unwrap_dimensional(wrapped, use_edgelist=use_edgelist, method=method)
    assert unwrapped.shape == (n, n)
    assert np.isnan(unwrapped).sum() == 16
    np.testing.assert_allclose(
        np.diff(unwrapped[:, :3], axis=1), np.diff(true_phase[:, :3], axis=1), atol=1e-9
    )
    np.testing.assert_allclose(
        np.diff(unwrapped[:, 5:], axis=1), np.diff(true_phase[:, 5:], axis=1), atol=1e-9
    )


def test_unwrap_dimensional_nan_start_pixel():
    true_phase = _ramp_2d()
    wrapped = wrap_difference(true_phase)
    wrapped[0, 0] = np.nan
    # start_pixel (0, 0) is NaN; unwrapper should fall back to first valid node
    unwrapped = unwrap_dimensional(wrapped, start_pixel=(0, 0))
    assert np.isnan(unwrapped[0, 0])
    assert not np.isnan(unwrapped[1:, 1:]).any()


def test_unwrap_dimensional_with_nan_weights_and_nans():
    true_phase = _ramp_2d()
    wrapped = wrap_difference(true_phase)
    wrapped[4:, :] = np.nan
    weights = np.ones(wrapped.shape)
    weights[0, 0] = np.nan  # NaN weight should become 0 weight on incident edges
    unwrapped = unwrap_dimensional(wrapped, weights=weights)
    assert np.isnan(unwrapped[4:, :]).all()
    assert not np.isnan(unwrapped[:4, :]).any()


def test_unwrap_dimensional_nan_grid_3d():
    z, y, x = np.mgrid[0:6, 0:6, 0:6]
    true_volume = 0.4 * x + 0.3 * y - 0.35 * z
    wrapped = wrap_difference(true_volume)
    wrapped[2:4, :, :] = np.nan

    unwrapped = unwrap_dimensional(wrapped)
    assert unwrapped.shape == wrapped.shape
    assert np.isnan(unwrapped).sum() == 2 * 6 * 6
    np.testing.assert_allclose(
        np.diff(unwrapped[:2], axis=2), np.diff(true_volume[:2], axis=2), atol=1e-9
    )
    np.testing.assert_allclose(
        np.diff(unwrapped[4:], axis=2), np.diff(true_volume[4:], axis=2), atol=1e-9
    )


@pytest.mark.parametrize("kwargs", [{}, {"use_edgelist": True}, {"method": "gc"}])
def test_unwrap_dimensional_all_nan(kwargs):
    res, info = unwrap_dimensional(np.full((6, 6), np.nan), return_info=True, **kwargs)
    assert np.isnan(res).all()
    assert info.success


def test_unwrap_dimensional_keeps_isolated_pixels():
    x = np.full((4, 4), np.nan)
    x[:2, :2] = 1.0  # 2x2 connected block
    x[3, 3] = 2.0  # finite pixel with no finite neighbour
    res = unwrap_dimensional(x)
    np.testing.assert_array_equal(res, x)


@pytest.mark.parametrize("kwargs", [{}, {"use_edgelist": True}, {"method": "gc"}])
def test_unwrap_dimensional_many_components(kwargs):
    # random NaNs split the grid into hundreds of components
    rng = np.random.default_rng(0)
    g = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    true_phase = 0.9 * g[0] + 0.7 * g[1]
    x = wrap_difference(true_phase)
    x[rng.random(x.shape) < 0.45] = np.nan
    edges, _ = kamui.get_2d_edges_and_simplices(x.shape)
    edges = edges[np.isfinite(x.ravel()[edges]).all(axis=1)]
    _, labels = connected_components(
        coo_matrix((np.ones(len(edges)), tuple(edges.T)), shape=(x.size,) * 2), directed=False
    )
    assert labels.max() > 100

    res = unwrap_dimensional(x, start_pixel=(1, 0), **kwargs).ravel()
    np.testing.assert_array_equal(np.isnan(res), np.isnan(x.ravel()))
    # every edge recovers the true difference, and every component keeps its
    # reference vertex: (1, 0) for its own, the lowest index for the others
    true_phase = true_phase.ravel()
    np.testing.assert_allclose(
        np.diff(res[edges], axis=1), np.diff(true_phase[edges], axis=1), atol=1e-9
    )
    roots = np.unique(labels, return_index=True)[1]
    roots[labels[40]] = 40
    np.testing.assert_allclose(res[roots], x.ravel()[roots])
