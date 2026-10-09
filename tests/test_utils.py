import numpy as np
import pytest

from kamui.utils import (
    _merge_weights,
    get_2d_edges_and_simplices,
    get_3d_edges_and_simplices,
    prepare_weights,
)


def _assert_valid_graph(edges, simplices, n_nodes):
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert edges.min() >= 0 and edges.max() < n_nodes
    directed = {tuple(e) for e in edges}
    for simplex in simplices:
        assert len(simplex) == 4
        assert all(0 <= v < n_nodes for v in simplex)
        for u, v in zip(simplex, simplex[1:] + simplex[:1]):
            assert (u, v) in directed or (v, u) in directed


def test_2d_counts_and_validity():
    edges, simplices = get_2d_edges_and_simplices((4, 5))
    assert len(edges) == 4 * 4 + 3 * 5
    assert len(simplices) == 3 * 4
    _assert_valid_graph(edges, simplices, 20)


def test_2d_cyclical_axis_adds_wrap_edges():
    plain_edges, _ = get_2d_edges_and_simplices((4, 5))
    edges, simplices = get_2d_edges_and_simplices((4, 5), cyclical_axis=0)
    assert len(edges) == len(plain_edges) + 5
    assert len(simplices) == 3 * 4 + 4
    _assert_valid_graph(edges, simplices, 20)


def test_2d_cyclical_axis_accepts_int_and_tuple():
    edges_int, _ = get_2d_edges_and_simplices((4, 5), cyclical_axis=1)
    edges_tuple, _ = get_2d_edges_and_simplices((4, 5), cyclical_axis=(1,))
    assert len(edges_int) == len(edges_tuple) == 4 * 4 + 3 * 5 + 4


def test_2d_short_axes_are_already_cyclical():
    edges, simplices = get_2d_edges_and_simplices((2, 5), cyclical_axis=(0, 1))
    # axis 0 has length 2, so only axis 1 contributes wrap edges
    assert len(edges) == 2 * 4 + 1 * 5 + 2
    _assert_valid_graph(edges, simplices, 10)


@pytest.mark.parametrize("cyclical_axis", [(), 0, 1, (0, 1)])
def test_2d_cells_share_edges_in_opposite_directions(cyclical_axis):
    # A consistent orientation makes the cycle matrix a network matrix, which
    # the min-cost-flow solver relies on: every edge on two cells must be
    # traversed forwards by one and backwards by the other.
    edges, simplices = get_2d_edges_and_simplices((5, 6), cyclical_axis=cyclical_axis)
    directions = {}
    for simplex in simplices:
        for u, v in zip(simplex, simplex[1:] + simplex[:1]):
            key, sign = ((u, v), 1) if u < v else ((v, u), -1)
            directions.setdefault(key, []).append(sign)
    shared = [signs for signs in directions.values() if len(signs) == 2]
    assert shared
    assert all(sorted(signs) == [-1, 1] for signs in shared)


def test_3d_counts_and_validity():
    edges, simplices = get_3d_edges_and_simplices((3, 4, 5))
    assert len(edges) == 2 * 4 * 5 + 3 * 3 * 5 + 3 * 4 * 4
    assert len(simplices) == 2 * 3 * 5 + 3 * 3 * 4 + 2 * 4 * 4
    _assert_valid_graph(edges, simplices, 60)


def test_3d_cyclical_axis_adds_wrap_edges():
    edges, simplices = get_3d_edges_and_simplices((3, 4, 5), cyclical_axis=2)
    assert len(edges) == 2 * 4 * 5 + 3 * 3 * 5 + 3 * 4 * 4 + 3 * 4
    assert len(simplices) == 2 * 3 * 5 + 3 * 3 * 4 + 2 * 4 * 4 + 2 * 4 + 3 * 3
    _assert_valid_graph(edges, simplices, 60)


def test_3d_short_axis_is_filtered():
    edges, _ = get_3d_edges_and_simplices((2, 4, 5), cyclical_axis=0)
    plain_edges, _ = get_3d_edges_and_simplices((2, 4, 5))
    assert len(edges) == len(plain_edges)


def test_merge_weights_uses_weights_as_given():
    weights = np.array([[0.0, 5.0], [10.0, 15.0]])
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    np.testing.assert_array_equal(_merge_weights(weights, edges, "sum"), [5, 10, 20, 25])
    np.testing.assert_array_equal(_merge_weights(weights, edges, "min"), [0, 0, 5, 10])
    np.testing.assert_array_equal(_merge_weights(weights, edges, "max"), [5, 10, 15, 15])


def test_merge_weights_keeps_integer_weights_integer_without_overflow():
    # no float cast: integers stay integers for LEMON, and NumPy widens
    # small integer types when summing
    weights = np.array([200, 100, 3], dtype=np.uint8)
    edges = np.array([[0, 1], [1, 2]])
    out = _merge_weights(weights, edges, "sum")
    assert np.issubdtype(out.dtype, np.integer)
    np.testing.assert_array_equal(out, [300, 103])


@pytest.mark.parametrize("merging_method", ["sum", "min", "max"])
def test_merge_weights_nan_becomes_zero(merging_method):
    weights = np.array([[np.nan, 1.0], [2.0, 3.0]])
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    out = _merge_weights(weights, edges, merging_method)
    np.testing.assert_array_equal(out[:2], 0)
    assert np.all(out[2:] > 0)


@pytest.mark.parametrize("merging_method", ["mean", "median"])
def test_merge_weights_rejects_bad_merging_method(merging_method):
    # "mean" is dropped: it unwraps like "sum" but turns odd integer sums into fractions
    with pytest.raises(ValueError, match="merging_method must be"):
        _merge_weights(np.ones((2, 2)), np.array([[0, 1]]), merging_method)


def test_prepare_weights_rescales_into_smoothing_range():
    weights = np.array([[0.0, 5.0], [10.0, 15.0]])
    edges, _ = get_2d_edges_and_simplices((2, 2))
    out = prepare_weights(weights, edges, smoothing=0.1, merging_method="mean")
    assert out.shape == (4,)
    assert out.min() >= 0.1
    assert out.max() <= 1.0


def test_prepare_weights_merging_methods_order():
    weights = np.array([[0.0, 1.0], [2.0, 3.0]])
    edges, _ = get_2d_edges_and_simplices((2, 2))
    w_min = prepare_weights(weights, edges, smoothing=0.0, merging_method="min")
    w_max = prepare_weights(weights, edges, smoothing=0.0, merging_method="max")
    w_mean = prepare_weights(weights, edges, smoothing=0.0, merging_method="mean")
    assert w_min.min() == pytest.approx(0.0)
    assert w_max.max() == pytest.approx(1.0)
    assert np.all(w_min <= w_mean) and np.all(w_mean <= w_max)


def test_prepare_weights_nan_becomes_zero():
    weights = np.array([[np.nan, 1.0], [2.0, 3.0]])
    edges, _ = get_2d_edges_and_simplices((2, 2))
    out = prepare_weights(weights, edges, smoothing=0.1, merging_method="min")
    assert not np.isnan(out).any()
    assert 0.0 in out


def test_prepare_weights_constant_input():
    weights = np.ones((2, 2))
    edges, _ = get_2d_edges_and_simplices((2, 2))
    out = prepare_weights(weights, edges)
    assert np.all(out == 1.0)


def test_prepare_weights_rejects_bad_smoothing():
    weights = np.ones((2, 2))
    edges, _ = get_2d_edges_and_simplices((2, 2))
    with pytest.raises(ValueError, match="smoothing"):
        prepare_weights(weights, edges, smoothing=1.0)
    with pytest.raises(ValueError, match="smoothing"):
        prepare_weights(weights, edges, smoothing=-0.1)


def test_prepare_weights_rejects_bad_merging_method():
    weights = np.ones((2, 2))
    edges, _ = get_2d_edges_and_simplices((2, 2))
    with pytest.raises(ValueError, match="min, max, mean"):
        prepare_weights(weights, edges, merging_method="median")
