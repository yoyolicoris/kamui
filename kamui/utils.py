"""Graph construction and weight preparation for phase unwrapping.

:func:`get_2d_edges_and_simplices` and :func:`get_3d_edges_and_simplices`
build the edge list and elementary 4-cycles of a regular grid (with
optional cyclical axes); :func:`prepare_weights` turns per-vertex quality
weights into per-edge weights.
"""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

__all__ = [
    "get_2d_edges_and_simplices",
    "get_3d_edges_and_simplices",
    "prepare_weights",
]


def get_2d_edges_and_simplices(
    shape: tuple[int, int], cyclical_axis: int | tuple[int, ...] = ()
) -> tuple[np.ndarray, Iterable[Iterable[int]]]:
    """Compute the edges and simplices for a 2-D grid.

    Parameters
    ----------
    shape : tuple of int
        The shape of the grid.
    cyclical_axis : int or tuple of int, optional
        The axis (or axes) treated as cyclical. Axes of length 2 or less
        are already cyclical and are ignored. Defaults to ().

    Returns
    -------
    edges : (M, 2) np.ndarray
        Array of edges, including wrap-around edges for cyclical axes.
    simplices : list of list of int
        Elementary 4-cycles of the grid, as vertex index lists.
    """
    nodes = np.arange(np.prod(shape)).reshape(shape)
    if isinstance(cyclical_axis, int):
        cyclical_axis = (cyclical_axis,)
    # if the axis length <= 2, then the axis is already cyclical
    cyclical_axis = tuple(filter(lambda ax: shape[ax] > 2, cyclical_axis))

    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1].ravel(), nodes[:, 1:].ravel()], axis=1),
            np.stack([nodes[:-1, :].ravel(), nodes[1:, :].ravel()], axis=1),
        )
        + tuple(
            np.stack(
                [
                    np.take(nodes, [0], axis=ax).ravel(),
                    np.take(nodes, [-1], axis=ax).ravel(),
                ],
                axis=1,
            )
            for ax in cyclical_axis
        ),
        axis=0,
    )
    simplices = np.stack(
        (
            nodes[:-1, :-1].ravel(),
            nodes[1:, :-1].ravel(),
            nodes[1:, 1:].ravel(),
            nodes[:-1, 1:].ravel(),
        ),
        axis=1,
    ).tolist()
    if len(cyclical_axis) > 0:
        pairs = []
        for ax in cyclical_axis:
            first = np.squeeze(np.take(nodes, [0], axis=ax), axis=ax)
            last = np.squeeze(np.take(nodes, [-1], axis=ax), axis=ax)
            # Orient the wrap-around cells like the interior ones, which step
            # down axis 0 first: across axis 0 that means starting from the
            # last row, so every edge two cells share is traversed in opposite
            # directions (as the min-cost-flow solver requires).
            pairs.append((last, first) if ax == 0 else (first, last))
        simplices += np.concatenate(
            tuple(
                np.stack(
                    (
                        x[:-1],
                        y[:-1],
                        y[1:],
                        x[1:],
                    ),
                    axis=1,
                )
                for x, y in pairs
            ),
            axis=0,
        ).tolist()
    return edges, simplices


def get_3d_edges_and_simplices(
    shape: tuple[int, int, int], cyclical_axis: int | tuple[int, ...] = ()
) -> tuple[np.ndarray, Iterable[Iterable[int]]]:
    """Compute the edges and simplices for a 3-D grid.

    Parameters
    ----------
    shape : tuple of int
        The shape of the grid.
    cyclical_axis : int or tuple of int, optional
        The axis (or axes) treated as cyclical. Axes of length 2 or less
        are already cyclical and are ignored. Defaults to ().

    Returns
    -------
    edges : (M, 2) np.ndarray
        Array of edges, including wrap-around edges for cyclical axes.
    simplices : list of list of int
        Elementary 4-cycles of the grid, as vertex index lists.
    """
    nodes = np.arange(np.prod(shape)).reshape(shape)
    if isinstance(cyclical_axis, int):
        cyclical_axis = (cyclical_axis,)
    cyclical_axis = tuple(filter(lambda ax: shape[ax] > 2, cyclical_axis))

    edges = np.concatenate(
        (
            np.stack([nodes[:, :-1, :].ravel(), nodes[:, 1:, :].ravel()], axis=1),
            np.stack([nodes[:-1, :, :].ravel(), nodes[1:, :, :].ravel()], axis=1),
            np.stack([nodes[:, :, :-1].ravel(), nodes[:, :, 1:].ravel()], axis=1),
        )
        + tuple(
            np.stack(
                [
                    np.take(nodes, [0], axis=ax).ravel(),
                    np.take(nodes, [-1], axis=ax).ravel(),
                ],
                axis=1,
            )
            for ax in cyclical_axis
        ),
        axis=0,
    )
    simplices = np.concatenate(
        (
            np.stack(
                (
                    nodes[:-1, :-1, :].ravel(),
                    nodes[1:, :-1, :].ravel(),
                    nodes[1:, 1:, :].ravel(),
                    nodes[:-1, 1:, :].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:, :-1, :-1].ravel(),
                    nodes[:, 1:, :-1].ravel(),
                    nodes[:, 1:, 1:].ravel(),
                    nodes[:, :-1, 1:].ravel(),
                ),
                axis=1,
            ),
            np.stack(
                (
                    nodes[:-1, :, :-1].ravel(),
                    nodes[:-1, :, 1:].ravel(),
                    nodes[1:, :, 1:].ravel(),
                    nodes[1:, :, :-1].ravel(),
                ),
                axis=1,
            ),
        ),
        axis=0,
    ).tolist()

    if len(cyclical_axis) > 0:
        simplices += np.concatenate(
            sum(
                (
                    (
                        np.stack(
                            (
                                x[1:, :].ravel(),
                                y[1:, :].ravel(),
                                y[:-1, :].ravel(),
                                x[:-1, :].ravel(),
                            ),
                            axis=1,
                        ),
                        np.stack(
                            (
                                x[:, 1:].ravel(),
                                y[:, 1:].ravel(),
                                y[:, :-1].ravel(),
                                x[:, :-1].ravel(),
                            ),
                            axis=1,
                        ),
                    )
                    for x, y in [
                        (
                            np.squeeze(np.take(nodes, [0], axis=ax), axis=ax),
                            np.squeeze(np.take(nodes, [-1], axis=ax), axis=ax),
                        )
                        for ax in cyclical_axis
                    ]
                ),
                (),
            ),
            axis=0,
        ).tolist()
    return edges, simplices


def _merge_weights(
    weights: npt.NDArray, edges: npt.NDArray[np.int_], merging_method: str
) -> npt.NDArray[np.floating]:
    """Combine the two vertex weights of each edge, as given, without rescaling.

    One of "sum", "min", "max" or "mean"; edges that touch a NaN weight get
    0. Scaling every weight by one factor leaves the unwrapping unchanged,
    and "sum", "min" and "max" keep integer weights integer for LEMON.
    """
    merge = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}.get(merging_method)
    if merge is None:
        raise ValueError(
            f"merging_method must be 'sum', 'min', 'max' or 'mean'; got {merging_method!r}"
        )
    edge_weights = merge(np.asarray(weights, dtype=np.float64).ravel()[edges], axis=1)
    edge_weights[np.isnan(edge_weights)] = 0
    return edge_weights


def prepare_weights(
    weights: npt.NDArray,
    edges: npt.NDArray[np.int_],
    smoothing: float = 0.1,
    merging_method: str = "mean",
) -> npt.NDArray[np.floating]:
    """Prepare per-edge weights from per-vertex weights.

    Assume the weights share the shape of the phase array to be unwrapped.
    Scale the weights from 0 to 1, pick the weights of the phase pairs
    connected by the edges, and merge each pair into one edge weight with
    ``merging_method``.

    Parameters
    ----------
    weights : np.ndarray
        Per-vertex weights, shaped like the phase array.
    edges : (M, 2) np.ndarray
        Edges connecting the phases.
    smoothing : float, optional
        Minimal rescaled value where weights are defined, in [0, 1).
        When positive, 0 is reserved for originally NaN weights; when 0,
        NaN weights and the smallest non-NaN ones both map to 0.
        Defaults to 0.1.
    merging_method : str, optional
        How to combine two phase weights into a single edge weight;
        one of "min", "max", "mean". Defaults to "mean".

    Returns
    -------
    (M,) np.ndarray
        Per-edge weights rescaled to [0, 1], with NaN entries replaced by 0.
    """
    if not 0 <= smoothing < 1:
        raise ValueError(
            "`smoothing` should be a value between 0 (inclusive) and 1 (non inclusive); got "
            + str(smoothing)
        )
    # scale the weights from 0 to 1
    weights = weights - np.nanmin(weights)
    current_max = np.nanmax(weights)
    if not current_max:
        # current maximum is 0, which means all weights originally had the same value,
        # now 0; replace everything with 1
        weights += 1
    else:
        weights /= current_max
        weights *= 1 - smoothing
        weights += smoothing
    # pick the weights corresponding to the phases connected by the edges
    # and use `merging_method` to get one weight for each edge
    allowed_merging_methods = ["min", "max", "mean"]
    if merging_method not in allowed_merging_methods:
        raise ValueError(
            "`merging_method` should be one of: "
            + ", ".join(allowed_merging_methods)
            + "; got "
            + str(merging_method)
        )
    weights_for_edges = getattr(np, merging_method)(weights.ravel()[edges], axis=1)

    # make sure there are no NaNs in the weights; replace any with 0s
    weights_for_edges[np.isnan(weights_for_edges)] = 0

    return weights_for_edges
