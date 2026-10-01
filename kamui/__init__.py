"""Robust and accurate phase unwrapping on 2-D, 3-D, or sparse data.

The public API has three entry points, arrays in / arrays out:

- :func:`wrap_difference`: wrap values into one period interval.
- :func:`unwrap_dimensional`: unwrap 2-D or 3-D phase arrays.
- :func:`unwrap_arbitrary`: unwrap phase on an arbitrary graph.

The ILP solvers (:func:`kamui.core.calculate_k`,
:func:`kamui.core.calculate_m`), the graph integrator
(:func:`kamui.core.integrate`), the PUMA graph-cut solver
(:func:`kamui.core.puma`), the grid builders
(:func:`kamui.utils.get_2d_edges_and_simplices`,
:func:`kamui.utils.get_3d_edges_and_simplices`), and the weight helper
(:func:`kamui.utils.prepare_weights`) are importable from here as well.
"""

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .core import calculate_k, calculate_m, integrate, puma
from .utils import (
    get_2d_edges_and_simplices,
    get_3d_edges_and_simplices,
    prepare_weights,
)

try:
    __version__ = version("kamui")
except PackageNotFoundError:  # pragma: no cover (source tree with no install)
    __version__ = "0.0.0+unknown"

__all__ = [
    "calculate_k",
    "calculate_m",
    "get_2d_edges_and_simplices",
    "get_3d_edges_and_simplices",
    "integrate",
    "prepare_weights",
    "puma",
    "unwrap_arbitrary",
    "unwrap_dimensional",
    "wrap_difference",
    "__version__",
]


def wrap_difference(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Wrap values into the interval ``[-period/2, period/2)``.

    Parameters
    ----------
    x : (N,) np.ndarray
        Input array.
    period : float, optional
        Length of the wrapping interval. Defaults to ``2 * np.pi``.

    Returns
    -------
    (N,) np.ndarray
        ``x`` wrapped into ``[-period/2, period/2)``.
    """
    return np.mod(x + period / 2, period) - period / 2


def unwrap_dimensional(
    x: np.ndarray,
    start_pixel: tuple[int, int] | tuple[int, int, int] | None = None,
    use_edgelist: bool = False,
    cyclical_axis: int | tuple[int, ...] = (),
    merging_method: str = "mean",
    weights: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray | None:
    """Unwrap the phase of a 2-D or 3-D array.

    Parameters
    ----------
    x : 2-D or 3-D np.ndarray
        The phase to be unwrapped.
    start_pixel : tuple of int, optional
        The reference pixel to start unwrapping.
        Defaults to (0, 0) for 2-D data and (0, 0, 0) for 3-D data.
    use_edgelist : bool, optional
        Whether to use the edgelist method. Defaults to False.
    cyclical_axis : int or tuple of int, optional
        The axis (or axes) treated as cyclical. Defaults to ().
    merging_method : str, optional
        Way of combining two phase weights into a single edge weight.
        Defaults to "mean".
    weights : np.ndarray, optional
        Weights defining the 'goodness' of value at each vertex.
        Shape must match the shape of x. Defaults to None.
    **kwargs
        Other arguments passed to :func:`kamui.unwrap_arbitrary`.

    Returns
    -------
    np.ndarray or None
        The unwrapped phase of the same shape as x, or None if the
        underlying solver reports infeasibility.
    """
    if start_pixel is None:
        start_pixel = (0,) * x.ndim
    assert x.ndim == len(start_pixel), "start_pixel must have the same dimension as x"

    start_i = 0
    for i, s in enumerate(start_pixel):
        start_i *= x.shape[i]
        start_i += s
    if x.ndim == 2:
        edges, simplices = get_2d_edges_and_simplices(x.shape, cyclical_axis=cyclical_axis)
    elif x.ndim == 3:
        edges, simplices = get_3d_edges_and_simplices(x.shape, cyclical_axis=cyclical_axis)
    else:
        raise ValueError("x must be 2D or 3D")
    psi = x.ravel()

    if weights is not None:
        # convert per-vertex weights to per-edge weights
        weights = prepare_weights(weights, edges=edges, merging_method=merging_method)
    result = unwrap_arbitrary(
        psi,
        edges,
        None if use_edgelist else simplices,
        start_i=start_i,
        weights=weights,
        **kwargs,
    )
    if result is None:
        return None
    return result.reshape(x.shape)


def unwrap_arbitrary(
    psi: np.ndarray,
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]] | None = None,
    method: str = "ilp",
    period: float = 2 * np.pi,
    start_i: int = 0,
    **kwargs: Any,
) -> np.ndarray | None:
    """Unwrap the phase of arbitrary data.

    Parameters
    ----------
    psi : 1-D np.ndarray of shape (P,)
        The phase (vertices) to be unwrapped.
    edges : 2-D np.ndarray of shape (M, 2)
        The edges of the graph.
    simplices : (N,) iterable of simplices, optional
        Each element is a list of vertices that form a simplex (a.k.a elementary cycle).
        The connections should be consistent with the edges.
        This is also used to compute automatic weights for each edge.
        If not provided and method is "ilp", an edgelist-based ILP solver
        will be used without weighting.
        Defaults to None.
    method : str, optional
        The method to be used. Valid options are "ilp" and "gc", where "gc" corresponds to PUMA.
        Defaults to "ilp".
    period : float, optional
        The period of the phase. Defaults to ``2 * np.pi``.
    start_i : int, optional
        The index of the reference vertex to start unwrapping. Defaults to 0.
    **kwargs
        Other arguments passed to the solver.

    Returns
    -------
    np.ndarray or None
        The unwrapped phase of the same shape as psi, or None if the
        underlying solver reports infeasibility.
    """
    if method == "gc":
        m = puma(psi / period, edges, **kwargs)
        m -= m[start_i]
        result = m * period + psi
    elif method == "ilp":
        if simplices is None:
            m = calculate_m(
                edges,
                np.round((psi[edges[:, 1]] - psi[edges[:, 0]]) / period).astype(np.int64),
                **kwargs,
            )
            if m is None:
                return None
            m -= m[start_i]
            result = m * period + psi
        else:
            diff = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], period)
            k = calculate_k(edges, simplices, diff / period, **kwargs)
            if k is None:
                return None
            correct_diff = diff + k * period

            result = (
                integrate(
                    np.concatenate((edges, np.flip(edges, 1)), axis=0),
                    np.concatenate((correct_diff, -correct_diff), axis=0),
                    start_i=start_i,
                )
                + psi[start_i]
            )
    else:
        raise ValueError("method must be 'gc' or 'ilp'")
    return result
