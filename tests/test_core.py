from types import SimpleNamespace

import numpy as np
import pytest

import kamui.core as core
from kamui.core import calculate_k, calculate_m, integrate, puma


def test_integrate_chain():
    # integrate is called with edges in both directions (as in
    # unwrap_arbitrary), so every vertex appears as a source
    edges = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
    weights = np.array([1.0, 2.0, 4.0, -1.0, -2.0, -4.0])
    np.testing.assert_allclose(integrate(edges, weights), [0.0, 1.0, 3.0, 7.0])


def _triangle():
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    simplices = [[0, 1, 2]]
    return edges, simplices


def test_calculate_k_recovers_loop_residue():
    edges, simplices = _triangle()
    differences = np.array([0.4, 0.4, 0.3])  # loop sum 1.1, so the residue is 1
    k = calculate_k(edges, simplices, differences, adaptive_weighting=False)
    assert k is not None
    assert k.sum() == -1
    assert np.abs(k).sum() == 1


def test_calculate_k_weight_options_agree_on_trivial_loop():
    edges, simplices = _triangle()
    differences = np.array([0.1, 0.1, -0.2])  # loop sum 0, so k is 0
    k_adaptive = calculate_k(edges, simplices, differences)
    k_uniform = calculate_k(edges, simplices, differences, adaptive_weighting=False)
    k_weighted = calculate_k(edges, simplices, differences, weights=np.ones(3))
    for k in (k_adaptive, k_uniform, k_weighted):
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


def test_calculate_k_returns_none_when_infeasible(monkeypatch):
    edges, simplices = _triangle()
    monkeypatch.setattr(core, "linprog", lambda *a, **k: SimpleNamespace(x=None))
    assert calculate_k(edges, simplices, np.array([0.1, 0.1, -0.2])) is None


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


def test_calculate_m_rejects_non_integer_differences():
    with pytest.raises(AssertionError):
        calculate_m(np.array([[0, 1]]), np.array([0.5]))


def test_calculate_m_returns_none_when_infeasible(monkeypatch):
    monkeypatch.setattr(core, "linprog", lambda *a, **k: SimpleNamespace(x=None))
    assert calculate_m(np.array([[0, 1]]), np.array([1], dtype=np.int64)) is None


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


def test_puma_requires_pymaxflow(monkeypatch):
    monkeypatch.setattr(core, "maxflow", None)
    with pytest.raises(ImportError, match="PyMaxflow"):
        puma(np.array([0.0, 0.5]), np.array([[0, 1]]))
