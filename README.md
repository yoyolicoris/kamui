# Kamui
[![CI](https://github.com/yoyolicoris/kamui/actions/workflows/ci.yml/badge.svg)](https://github.com/yoyolicoris/kamui/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/kamui.svg)](https://badge.fury.io/py/kamui)


Kamui is a python package for robust and accurate phase unwrapping on 2-D, 3-D, or sparse data. 

## How it works

Kamui views the data as a graph $`G = (V, E)`$, with the wrapped phase $`\psi_v \in [-\pi, \pi)`$ at each vertex $`v`$. Along each edge $`e = (u, v)`$, the wrapped phase difference is

```math
x_e = \mathcal{W}(\psi_v - \psi_u),
\qquad
\mathcal{W}(\theta) = \theta - 2\pi \left\lfloor \frac{\theta + \pi}{2\pi} \right\rfloor \in [-\pi, \pi).
```

The unwrapped phase $`\phi`$ differs from $`\psi`$ by whole multiples of $`2\pi`$, so its differences are $`\phi_v - \phi_u = x_e + 2\pi k_e`$, for an integer ambiguity $`k_e`$ on each edge.
True phase differences are irrotational: they sum to zero around every elementary cycle of the graph, such as a grid cell or a mesh triangle.
Kamui picks the cheapest ambiguities that satisfy this by solving the integer linear program (ILP)

```math
\min_{k \in \mathbb{Z}^{M}} \; \sum_{e \in E} w_e \lvert k_e \rvert
\quad \text{s.t.} \quad
A k = -\frac{1}{2\pi} A x .
```

Here:

- $`M = \lvert E \rvert`$ is the number of edges, and $`w \in \mathbb{R}_{\ge 0}^{M}`$ are the edge weights.
- $`A \in \{-1, 0, 1\}^{N \times M}`$ is the incidence matrix of the $`N`$ elementary cycles. $`A_{ce} = 1`$ if cycle $`c`$ traverses edge $`e`$ from $`u`$ to $`v`$, $`-1`$ if it traverses it from $`v`$ to $`u`$, and $`0`$ if $`e`$ is not on $`c`$.
- The right-hand side is integral: $`\frac{1}{2\pi} (A x)_c`$ is the residue of cycle $`c`$, the number of whole turns that the wrapped differences around it add up to.

Kamui then integrates $`x + 2\pi k`$ along a spanning tree from the reference vertex to recover $`\phi`$.
This is the general form of the network programming approach proposed in the paper "[A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)".
With `period=T`, replace $`2\pi`$ by $`T`$ throughout.

On 2-D grids and planar meshes, every edge lies on at most two cycles. Kamui orients the cycles so that they traverse each shared edge in opposite directions, reversing any that do not; reversing a cycle negates both sides of its constraint, so the program does not change. $`A`$ is then the incidence matrix of the dual graph, which has one node per cycle and one more for the outside, and the program is a min-cost flow: the residues are the supplies and $`k_e`$ is the flow across edge $`e`$.
Kamui solves these programs with the network simplex of the [LEMON](https://lemon.cs.elte.hu/) graph library, which is many times faster than a general LP solver. LEMON needs integer weights; see [Weights](#weights).

Otherwise, as on 3-D grids or with fractional weights, Kamui first solves the linear programming (LP) relaxation, with $`k \in \mathbb{R}^{M}`$, using HiGHS.
For 2-D grids and planar meshes, $`A`$ is totally unimodular, so the LP optimum is already integral. The same is true of 3-D grids without a cyclical axis. Other inputs fall back to the integer program.
Large inputs are still computationally heavy; see [Performance and memory](#performance-and-memory).

## Installation

```commandline
pip install kamui
```

This also installs [pylmcf](https://github.com/michalsta/pylmcf), Python bindings for LEMON's network simplex under the Boost Software License, which Kamui uses wherever it applies.

Kamui also provides [PUMA](https://ieeexplore.ieee.org/document/4099386), a fast and robust phase unwrapping algorithm based on graph cuts as an alternative.
To install PUMA, run

```commandline
pip install kamui[extra]
```

However, it uses the original maxflow implementation by Vladimir Kolmogorov with GPL license.
Please follow the licensing instruction in [PyMaxflow](http://pmneila.github.io/PyMaxflow/#indices-and-tables) if you use this version of Kamui.


## Usage

The examples below build on each other and run as written.

### 2-D data

```python
import numpy as np
import kamui

# a smooth synthetic phase surface, wrapped into [-pi, pi)
y, x = np.mgrid[0:128, 0:128]
true_phase = 0.002 * (x - 40) ** 2 + 0.001 * (y - 70) ** 2 + 0.05 * x
wrapped = kamui.wrap_difference(true_phase)

unwrapped = kamui.unwrap_dimensional(wrapped)

# the result equals the true phase up to a constant multiple of 2 pi
assert np.allclose(unwrapped - unwrapped[0, 0], true_phase - true_phase[0, 0])
```

The reference pixel keeps its wrapped value. It defaults to the first pixel; choose another with `start_pixel`, for example `start_pixel=(64, 64)`.

### Weights

Per-pixel quality weights, such as InSAR coherence, tell the solver where corrections are cheap. Each edge gets the sum of its two pixels' weights; `merging_method="min"` or `"max"` change that. Edges that touch a NaN weight get weight 0. Kamui uses the weights as given: multiplying them all by one factor does not change the result.

LEMON needs integer weights, and Kamui does not round them for you. Fractional weights are solved by HiGHS, which is several times slower. To use LEMON, scale and round the weights yourself; a larger factor keeps more resolution:

```python
coherence = np.random.default_rng(0).uniform(0.2, 1.0, wrapped.shape)
unwrapped = kamui.unwrap_dimensional(wrapped, weights=np.round(coherence * 100))
```

With `solver="lemon"`, fractional weights raise an error instead of falling back to HiGHS.

`unwrap_arbitrary` takes per-edge weights and uses them exactly as given. Up to version 0.2, `unwrap_dimensional` rescaled the per-pixel weights linearly to $`[0.1, 1]`$ before merging them. `kamui.prepare_weights` still does that, so pass its result to `unwrap_arbitrary` to keep that behaviour:

```python
edges, cycles = kamui.get_2d_edges_and_simplices(wrapped.shape)
edge_weights = kamui.prepare_weights(coherence, edges)  # rescaled, then averaged per edge
unwrapped = kamui.unwrap_arbitrary(wrapped.ravel(), edges, cycles, weights=edge_weights)
unwrapped = unwrapped.reshape(wrapped.shape)
```

### Cyclical axes

For data that wraps around along an axis, such as the angle of a polar grid, mark that axis as cyclical so the first and last slices are connected:

```python
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
radius = np.linspace(1, 3, 48)
true_ring = 6 * np.sin(theta)[:, None] + 2.5 * radius[None, :]

unwrapped = kamui.unwrap_dimensional(kamui.wrap_difference(true_ring), cyclical_axis=0)
assert np.allclose(unwrapped - unwrapped[0, 0], true_ring - true_ring[0, 0])
```

### 3-D data

```python
z, y, x = np.mgrid[0:16, 0:16, 0:16]
true_volume = 0.4 * x + 0.3 * y - 0.35 * z + 0.01 * (x - 8) ** 2

unwrapped = kamui.unwrap_dimensional(kamui.wrap_difference(true_volume))
assert np.allclose(unwrapped - unwrapped[0, 0, 0], true_volume - true_volume[0, 0, 0])
```

### Point clouds and other graphs

For scattered points, use `unwrap_arbitrary` with the phase at each point, the edges between points and, optionally, the elementary cycles ("simplices"), such as the triangles of a Delaunay triangulation:

```python
from scipy.spatial import Delaunay

rng = np.random.default_rng(0)
points = rng.uniform(0, 10, (2000, 2))
point_phase = 1.5 * points[:, 0] - points[:, 1]
psi = kamui.wrap_difference(point_phase)

triangles = Delaunay(points).simplices
# Delaunay adds long, thin triangles along the convex hull. Drop them, so that
# no edge is too long for its phase change to be resolved.
sides = triangles[:, [[0, 1], [1, 2], [2, 0]]]
lengths = np.linalg.norm(points[sides[..., 0]] - points[sides[..., 1]], axis=-1)
keep = lengths.max(axis=1) < 1.0
triangles = triangles[keep]
edges = np.unique(np.sort(sides[keep].reshape(-1, 2), axis=1), axis=0)

unwrapped = kamui.unwrap_arbitrary(psi, edges, triangles)
assert np.allclose(unwrapped - unwrapped[0], point_phase - point_phase[0])
```

Without `simplices`, `unwrap_arbitrary(psi, edges)` uses the edgelist formulation, which needs only edges. That suits graphs without a natural cycle structure, such as interferogram networks in time series analysis.

### Choosing a solver

| Solver | How to select it | Needs | Notes |
| --- | --- | --- | --- |
| Simplex ILP (default) | `unwrap_dimensional(x)`, or `unwrap_arbitrary(psi, edges, simplices)` | edges and elementary cycles | By default each edge costs the number of its cycles that have no residue; `weights` overrides that. 2-D grids and planar meshes with integer weights, the defaults included, are solved by LEMON, and everything else by HiGHS. `solver="highs"` forces HiGHS, and `solver="lemon"` raises an error instead of falling back. |
| Edgelist ILP | `use_edgelist=True`, or `unwrap_arbitrary(psi, edges)` | edges only | Uniform weights unless `weights` is given. Integer weights, the default included, are solved by LEMON, and fractional ones by HiGHS; `solver` works as for the simplex ILP. |
| PUMA | `method="gc"` | edges and `pip install kamui[extra]` | Minimizes a $`p`$-norm energy (options `p` and `max_jump`) and takes no weights. Currently slower than the ILP solvers on grids ([#26](https://github.com/yoyolicoris/kamui/issues/26)). GPL; see [Installation](#installation). |

The edgelist ILP needs no cycles, because it optimizes vertex offsets $`m \in \mathbb{Z}^{\lvert V \rvert}`$ directly, with $`\phi = \psi + 2\pi m`$:

```math
\min_{m \in \mathbb{Z}^{\lvert V \rvert}} \; \sum_{e = (u, v) \in E} w_e \left\lvert m_v - m_u + \text{round}\left( \frac{\psi_v - \psi_u}{2\pi} \right) \right\rvert .
```

The term inside the absolute value is the edge ambiguity $`k_e`$. This is therefore the cost of the simplex ILP, optimized over offsets instead of ambiguities. The two optima agree whenever the cycles cover every loop of the graph, as grid cells and mesh triangles do.

Its LP dual is a min-cost circulation on the graph itself, with a flow $`y_e \in [-w_e, w_e]`$ along each edge that maximizes $`\sum_e y_e \, \text{round}\left( (\psi_v - \psi_u) / 2\pi \right)`$. LEMON solves that circulation, and its node potentials are an optimal $`m`$.

PUMA instead uses graph cuts to minimize the $`p`$-norm of the unwrapped differences:

```math
\min_{m \in \mathbb{Z}^{\lvert V \rvert}} \; \sum_{e = (u, v) \in E} \lvert \phi_v - \phi_u \rvert^{p},
\qquad \phi = \psi + 2\pi m .
```

### Solver report

Pass `return_info=True` to also get the solver's report, a [`scipy.optimize.OptimizeResult`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html). It is returned even when the result is `None`, so `info.message` says why the solve failed.

```python
unwrapped, info = kamui.unwrap_dimensional(wrapped, return_info=True)
print(info.success, info.fun, info.ilp_fallback)
```

- `fun` is the weighted L1 cost of the corrections; for PUMA, it is the final energy.
- `success`, `status` and `message` come from the solver.
- `solver` names the solver that ran: `"lemon"` or `"highs"`.
- `ilp_fallback` tells whether the LP optimum was fractional, so HiGHS solved the integer program instead.

With the same weights, the simplex and edgelist solvers minimize the same cost, so their `fun` values can be compared directly:

```python
noisy = kamui.wrap_difference(true_phase + np.random.default_rng(1).normal(0, 1.0, wrapped.shape))
edges, cycles = kamui.get_2d_edges_and_simplices(noisy.shape)

_, simplex = kamui.unwrap_arbitrary(
    noisy.ravel(), edges, cycles, weights=np.ones(len(edges)), return_info=True
)
_, edgelist = kamui.unwrap_arbitrary(noisy.ravel(), edges, return_info=True)
assert np.isclose(simplex.fun, edgelist.fun)
```

### Missing data

Kamui expects a single connected graph with finite phase values. NaNs, or regions cut off from the reference point, currently give wrong results ([#23](https://github.com/yoyolicoris/kamui/issues/23)). Until that is fixed, drop invalid pixels and unwrap each connected region on its own with `unwrap_arbitrary`.

## Performance and memory

`unwrap_dimensional` with default settings on noisy 2-D grids, measured on an Apple M1 Pro with SciPy 1.18 (HiGHS 1.12) and pylmcf 1.2.1:

| Grid | HiGHS | LEMON |
| --- | --- | --- |
| 300×300 | 1.6 s, 0.7 GB | 0.35 s, 0.3 GB |
| 600×600 | 9.8 s, 1.7 GB | 1.6 s, 0.8 GB |
| 1000×1000 | 57 s, 2.6 GB | 5.0 s, 2.2 GB |
| 2000×2000 | — | 32 s |

The edgelist path (`use_edgelist=True`) and grids with integer weights gain 3–4× from LEMON: at 600×600, from 78 s to 18 s and from 14 s to 4.3 s.

Both solvers reach the same optimal cost. With LEMON, the solve is no longer the bottleneck: on large grids, most of the time and memory go into Kamui's own Python code that builds the program, which [#24](https://github.com/yoyolicoris/kamui/issues/24) addresses. Scenes such as the 4628×2562 interferograms in [#11](https://github.com/yoyolicoris/kamui/issues/11) and [#12](https://github.com/yoyolicoris/kamui/issues/12) still need tens of GB of memory.

If unwrapping is unexpectedly slow, check your SciPy build. On macOS (Apple Silicon), conda-forge's SciPy 1.15.0–1.15.2 builds solve kamui's programs hundreds of times slower than SciPy's PyPI wheels and other conda-forge releases, for example 5 s instead of 10 ms for a 32×32 grid.

## Examples

- [Unwrapping on 2-D synthetic data](https://github.com/yoyolicoris/kamui/blob/dev/examples/synthetic_images.ipynb)

## Roadmap

- [ ] Missing data and disconnected graphs ([#23](https://github.com/yoyolicoris/kamui/issues/23))
- [ ] Lower memory use on large scenes ([#24](https://github.com/yoyolicoris/kamui/issues/24))
- [x] A min-cost-flow solver for 2-D and planar data ([#25](https://github.com/yoyolicoris/kamui/issues/25))
- [ ] Faster PUMA ([#26](https://github.com/yoyolicoris/kamui/issues/26))
- [ ] A conda-forge package ([#13](https://github.com/yoyolicoris/kamui/issues/13))

## References

- [scikit-image/scikit-image/#4622](https://github.com/scikit-image/scikit-image/issues/4622)
- [my medium blogpost](https://medium.com/@ILoveJK/%E7%9B%B8%E4%BD%8D%E9%87%8D%E5%BB%BA%E8%88%87%E5%9C%96%E5%AD%B8-phase-unwrapping-using-minimum-cost-network-flow-%E4%B8%89-b64732901f17)
- [A novel phase unwrapping method based on network programming](https://ieeexplore.ieee.org/document/673674)
- [Phase Unwrapping via Graph Cuts](https://ieeexplore.ieee.org/document/4099386)
- [Edgelist phase unwrapping algorithm for time series InSAR analysis](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-27-3-605)
- [Time Series Phase Unwrapping Based on Graph Theory and Compressed Sensing](https://ieeexplore.ieee.org/document/9387451?arnumber=9387451)
