# Kamui
[![Lint](https://github.com/yoyololicon/kamui/actions/workflows/black.yml/badge.svg)](https://github.com/yoyololicon/kamui/actions/workflows/black.yml)
[![Upload Python Package](https://github.com/yoyololicon/kamui/actions/workflows/python-publish.yml/badge.svg)](https://github.com/yoyololicon/kamui/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/kamui.svg)](https://badge.fury.io/py/kamui)


Kamui is a python package for robust and accurate phase unwrapping on 2-D, 3-D, or sparse data. 

Kamui unwrap the phases by viewing the data points as vertices $V$ connected with edges $E$ and solving the following integer linear programming (ILP) problem:

```math
\min_{k} w^T |k|,
```

```math
\text{s.t.} Ak = -A\frac{x}{2\pi},
```
where $`k_{i \in [0, M)} \in \mathbb{Z}`$ is the edge ambiguities to be computed, $`w_{i \in [0, M)} \in \mathbb{R}^+`$ is the weights, $`x_{i \in [0, M)} = (V_v - V_u + \pi) \pmod {2\pi} - \pi |  (u, v) = E_i`$ is the pseudo phase derivatives, $`M = |E|`$. 
$`A_{ij} \in \{-1, 0, 1\} | i \in [0, N) \cap j \in [0, M)`$ and $N$ is the number of elementary cycles enclosed by $E$.

This formulation is based on the fact that the true phase differences, $2\pi k + x$, should fulfill the irrotationality constraint, which means the summation of phase derivatives of each elementary cycles is zero.
This is the general form of the network programming approach proposed in the paper "[A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)".

Unwrapping phase with Kamui can be computationally heavy.
Acceleration techniques, such as dividing the graph into subgraphs, will be implemented in the future.

## Installation

```commandline
pip install kamui
```

Kamui also provides [PUMA](https://ieeexplore.ieee.org/document/4099386), a fast and robust phase unwrapping algorithm based on graph cuts as an alternative.
To install PUMA, run

```commandline
pip install kamui[extra]
```

However, it uses the original maxflow implementation by Vladimir Kolmogorov with GPL license.
Please follow the licensing instruction in [PyMaxflow](http://pmneila.github.io/PyMaxflow/#indices-and-tables) if you use this version of Kamui.


## Usage

For regular 2-D or 3-D data such as interferograms, use `kamui.unwrap_dimensional`:

```python
import numpy as np

def unwrap_dimensional(
    x: np.ndarray,
    start_pixel: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    use_edgelist: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Unwrap the phase of a 2-D or 3-D array.

    Parameters
    ----------
    x : 2-D or 3-D np.ndarray
        The phase to be unwrapped.
    start_pixel : (2,) or (3,) tuple
        the reference pixel to start unwrapping.
        Default to (0, 0) for 2-D data and (0, 0, 0) for 3-D data.
    use_edgelist : bool
        Whether to use the edgelist method.
        Default to False.
    kwargs : dict
        Other arguments passed to `kamui.unwrap_arbitrary`.

    Returns
    -------
    np.ndarray
        The unwrapped phase of the same shape as x.
    """
```

For sparse data, use `kamui.unwrap_arbitrary`:

```python
import numpy as np

def unwrap_arbitrary(
    psi: np.ndarray,
    edges: np.ndarray,
    simplices: Iterable[Iterable[int]] = None,
    method: str = "ilp",
    period: float = 2 * np.pi,
    start_i: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Unwrap the phase of arbitrary data.

    Parameters
    ----------
    psi : 1D np.ndarray of shape (P,)
        The phase (vertices) to be unwrapped.
    edges : 2-D np.ndarray of shape (M, 2)
        The edges of the graph.
    simplices : Iterable[Iterable[int]] of length (N,)
        Each element is a list of vertices that form a simplex (a.k.a elementary cycle).
        The connections should be consistent with the edges.
        This is also used to compute automatic weights for each edge.
        If not provided and method is "ilp", an edgelist-based ILP solver will be used without weighting.
    method : str
        The method to be used. Valid options are "ilp" and "gc", where "gc" correponds to PUMA.
        Default to "ilp".
    period : float
        The period of the phase.
        Default to 2 * np.pi.
    start_i : int
        The index of the reference vertex to start unwrapping.
        Default to 0.
    kwargs : dict
        Other arguments passed to the solver.

    Returns
    -------
    np.ndarray
        The unwrapped phase of the same shape as psi.
    """
```

## Examples

- [Unwrapping on 2-D synthetic data](/examples/synthetic_images.ipynb)

## TODO

- [ ] subgraph division
- [x] edges-based custom weighting
- [ ] vertices-based custom weighting

## References

- [scikit-image/scikit-image/#4622](https://github.com/scikit-image/scikit-image/issues/4622)
- [my medium blogpost](https://medium.com/@ILoveJK/%E7%9B%B8%E4%BD%8D%E9%87%8D%E5%BB%BA%E8%88%87%E5%9C%96%E5%AD%B8-phase-unwrapping-using-minimum-cost-network-flow-%E4%B8%89-b64732901f17)
- [A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)
- [Phase Unwrapping via Graph Cuts](https://ieeexplore.ieee.org/document/4099386)
- [Edgelist phase unwrapping algorithm for time series InSAR analysis](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-27-3-605)
- [Time Series Phase Unwrapping Based on Graph Theory and Compressed Sensing](https://ieeexplore.ieee.org/document/9387451?arnumber=9387451)
