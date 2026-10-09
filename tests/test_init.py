import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.spatial import Delaunay

import kamui
from kamui import (
    get_2d_edges_and_simplices,
    unwrap_arbitrary,
    unwrap_dimensional,
    wrap_difference,
)


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
    true = _ramp_2d()
    result = unwrap_dimensional(wrap_difference(true))
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true - true[0, 0], atol=1e-6)


def test_unwrap_dimensional_edgelist():
    true = _ramp_2d(5, 5)
    result = unwrap_dimensional(wrap_difference(true), use_edgelist=True)
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true - true[0, 0], atol=1e-6)


def test_unwrap_dimensional_weights_and_start_pixel():
    true = _ramp_2d()
    weights = np.linspace(0.5, 1.0, true.size).reshape(true.shape)
    result = unwrap_dimensional(wrap_difference(true), start_pixel=(2, 3), weights=weights)
    assert result is not None
    np.testing.assert_allclose(result - result[2, 3], true - true[2, 3], atol=1e-6)


@pytest.mark.parametrize("use_edgelist", [False, True])
def test_unwrap_dimensional_integer_weights_reach_lemon(use_edgelist):
    rng = np.random.default_rng(4)
    true = _ramp_2d(12, 12) + rng.normal(0, 1.3, (12, 12))
    coherence = rng.uniform(0.2, 1.0, true.shape)
    wrapped = wrap_difference(true)
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
    true = 2.5 * np.sin(2 * np.pi * ii / n) + 0.4 * jj
    result = unwrap_dimensional(wrap_difference(true), cyclical_axis=0)
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true - true[0, 0], atol=1e-6)


def test_unwrap_dimensional_3d():
    n = 3
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    true = 1.2 * (xx + yy + zz)
    result = unwrap_dimensional(wrap_difference(true))
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0, 0], true - true[0, 0, 0], atol=1e-6)


def test_unwrap_dimensional_gc():
    true = _ramp_2d()
    result = unwrap_dimensional(wrap_difference(true), method="gc")
    assert result is not None
    np.testing.assert_allclose(result - result[0, 0], true - true[0, 0], atol=1e-6)


def test_unwrap_dimensional_3d_ramp_on_larger_grid():
    # 10x10x10 is large enough for consecutive voxels in the traversal order
    # not to be adjacent, which broke integrate before #22
    zz, yy, xx = np.mgrid[0:10, 0:10, 0:10]
    true = 1.3 * zz + 0.9 * yy - 1.1 * xx
    result = unwrap_dimensional(wrap_difference(true))
    np.testing.assert_allclose(result - result[0, 0, 0], true - true[0, 0, 0], atol=1e-6)


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
    true = np.array([0.0, 2.0, 4.0, 6.0])
    psi = wrap_difference(true)
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    result = unwrap_arbitrary(psi, edges, None)
    np.testing.assert_allclose(result, true, atol=1e-9)


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
        true = 3 * p[:, 0] - 2 * p[:, 1]
        psi = wrap_difference(true)
        edges, simplices = _delaunay_graph(p)
        result = unwrap_arbitrary(psi, edges, simplices)
        np.testing.assert_allclose(result - result[0], true - true[0], atol=1e-6)


def test_unwrap_arbitrary_reports_matching_costs_on_both_ilp_paths():
    # With uniform weights, the simplex and edgelist programs minimize the
    # same cost over the same corrections, so their optima must agree
    # (the comparison asked for in #9).
    rng = np.random.default_rng(3)
    true = _ramp_2d(12, 12) + rng.normal(0, 1.3, (12, 12))
    psi = wrap_difference(true).ravel()
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
    true = _ramp_2d()
    result, info = unwrap_dimensional(wrap_difference(true), return_info=True)
    assert result.shape == true.shape
    assert info.success and not info.ilp_fallback
    np.testing.assert_allclose(result - result[0, 0], true - true[0, 0], atol=1e-6)


def test_unwrap_dimensional_reports_failure(monkeypatch):
    failed = OptimizeResult(success=False)
    monkeypatch.setattr(kamui, "unwrap_arbitrary", lambda *a, **k: (None, failed))
    assert unwrap_dimensional(np.zeros((4, 4)), return_info=True) == (None, failed)


def test_unwrap_arbitrary_gc():
    psi = np.array([0.0, 1.0])
    edges = np.array([[0, 1]])
    result = unwrap_arbitrary(psi, edges, method="gc")
    np.testing.assert_allclose(result, psi, atol=1e-9)


def test_unwrap_arbitrary_rejects_bad_method():
    with pytest.raises(ValueError, match="method must be"):
        unwrap_arbitrary(np.zeros(2), np.array([[0, 1]]), method="bad")
