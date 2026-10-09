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


def prepare_weights(
    weights: npt.NDArray,
    edges: npt.NDArray[np.int_],
    merging_method: str = "sum",
) -> npt.NDArray[np.floating]:
    """Prepare per-edge weights from per-vertex weights.

    Each edge combines the weights of its two vertices with
    ``merging_method``. The weights are used as given: scaling them all by
    one factor leaves the unwrapping unchanged, but shifting them does not,
    so kamui does not rescale them.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative per-vertex weights, shaped like the phase array; NaN
        marks vertices without a weight.
    edges : (M, 2) np.ndarray
        Edges connecting the phases.
    merging_method : str, optional
        How to combine the two vertex weights into one edge weight; one of
        "sum", "min", "max", "mean". "sum" and "mean" give the same
        unwrapping, but only "sum", "min" and "max" keep integer weights
        integer, which LEMON needs. Defaults to "sum".

    Returns
    -------
    (M,) np.ndarray
        Per-edge weights; edges that touch a NaN weight get 0.

    Raises
    ------
    ValueError
        If ``merging_method`` is not one of the above.
    """
    merge = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}.get(merging_method)
    if merge is None:
        raise ValueError(
            f"merging_method must be 'sum', 'min', 'max' or 'mean'; got {merging_method!r}"
        )
    edge_weights = merge(np.asarray(weights, dtype=np.float64).ravel()[edges], axis=1)
    edge_weights[np.isnan(edge_weights)] = 0
    return edge_weights
