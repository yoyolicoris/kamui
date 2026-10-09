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
from scipy.optimize import OptimizeResult

from .core import calculate_k, calculate_m, integrate, puma
from .utils import (
    _merge_weights,
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
    merging_method: str = "sum",
    weights: np.ndarray | None = None,
    *,
    return_info: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray | None, OptimizeResult] | None:
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
        How to combine the weights of an edge's two pixels: "sum", "min" or
        "max". Defaults to "sum".
    weights : np.ndarray, optional
        Non-negative per-pixel weights defining the 'goodness' of each
        value, shaped like x; NaN marks pixels without a weight, and edges
        that touch one get weight 0. They are used as given, without
        rescaling. Only the ILP solvers (``method="ilp"``) use weights.
        Integer weights, such as ``np.round(coherence * 100)``, let LEMON
        solve the program; fractional ones go to HiGHS. For the rescaled
        weights of :func:`kamui.prepare_weights`, pass its result to
        :func:`kamui.unwrap_arbitrary` instead. Defaults to None.
    return_info : bool, optional
        Also return the solver report. Defaults to False.
    **kwargs
        Other arguments passed to :func:`kamui.unwrap_arbitrary`.

    Returns
    -------
    unwrapped : np.ndarray or None
        The unwrapped phase of the same shape as x, or None if the
        underlying solver finds no optimal solution.
    info : scipy.optimize.OptimizeResult
        Only returned if ``return_info`` is True; see
        :func:`kamui.unwrap_arbitrary`.
    """
    if start_pixel is None:
        start_pixel = (0,) * x.ndim
    if len(start_pixel) != x.ndim:
        raise ValueError(
            f"start_pixel needs one index per dimension of x: got {len(start_pixel)} "
            f"for a {x.ndim}-D array"
        )

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
        # convert per-vertex weights to per-edge weights; forward them only
        # when given, since puma (method="gc") takes no weights argument
        kwargs["weights"] = _merge_weights(weights, edges, merging_method)
    out = unwrap_arbitrary(
        psi,
        edges,
        None if use_edgelist else simplices,
        start_i=start_i,
        return_info=return_info,
        **kwargs,
    )
    result, info = out if return_info else (out, None)
    if result is not None:
        result = result.reshape(x.shape)
    return (result, info) if return_info else result


def unwrap_arbitrary(
    psi: np.ndarray,
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]] | None = None,
    method: str = "ilp",
    period: float = 2 * np.pi,
    start_i: int = 0,
    *,
    return_info: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray | None, OptimizeResult] | None:
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
    return_info : bool, optional
        Also return the solver report. Defaults to False.
    **kwargs
        Other arguments passed to the solver.

    Returns
    -------
    unwrapped : np.ndarray or None
        The unwrapped phase of the same shape as psi, or None if the
        underlying solver finds no optimal solution.
    info : scipy.optimize.OptimizeResult
        Only returned if ``return_info`` is True. For ``method="ilp"`` it
        is the report of :func:`kamui.core.calculate_k` (with simplices) or
        :func:`kamui.core.calculate_m` (without): ``fun`` is the weighted
        L1 cost, ``success``, ``status`` and ``message`` come from the
        solver, which ``solver`` names ("lemon" or "highs"; pass
        ``solver=`` through ``**kwargs`` to choose it), and
        ``ilp_fallback`` tells whether HiGHS had to solve the integer
        program. For ``method="gc"`` it is the report of
        :func:`kamui.core.puma`, with the final energy as ``fun``.
    """
    if method == "gc":
        m, info = puma(psi / period, edges, return_info=True, **kwargs)
        result = (m - m[start_i]) * period + psi
    elif method == "ilp":
        if simplices is None:
            m, info = calculate_m(
                edges,
                np.round((psi[edges[:, 1]] - psi[edges[:, 0]]) / period).astype(np.int64),
                return_info=True,
                **kwargs,
            )
            result = None if m is None else (m - m[start_i]) * period + psi
        else:
            diff = wrap_difference(psi[edges[:, 1]] - psi[edges[:, 0]], period)
            k, info = calculate_k(edges, simplices, diff / period, return_info=True, **kwargs)
            if k is None:
                result = None
            else:
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
    return (result, info) if return_info else result
