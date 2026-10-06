# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The environment is managed by Pixi (`pixi.toml`, `pixi.lock`). The default environment is Python 3.13 plus test and dev tooling. `py310` through `py314` are test-only environments, and CI runs every one of them on Linux, macOS and Windows.

```sh
pixi run test            # pytest with coverage; fails below 100% (fail_under in pyproject.toml)
pixi run lint            # ruff check kamui tests scripts
pixi run format          # ruff format (format-check is the CI variant)
pixi run docstrings      # numpydoc validation of the public API in kamui/*.py
pixi run build           # sdist + wheel
pixi run smoke           # install the built wheel in an isolated venv and check its version
pixi run -e py310 test   # the suite on another interpreter
```

Run a single test with plain pytest. The `test` task would fail the coverage threshold on a subset.

```sh
pixi run pytest tests/test_core.py::test_name -v
```

CI runs Pixi with `locked: true`. After editing `pixi.toml`, run `pixi install` and commit the updated `pixi.lock`.

## Architecture

The package has three modules.

- **`kamui/__init__.py`** holds the public entry points.
  - `unwrap_dimensional` builds the grid graph, turns per-pixel `weights` into per-edge weights, and hands off to `unwrap_arbitrary`.
  - `unwrap_arbitrary` picks the solver:
    - `method="ilp"` with simplices: the simplex ILP (`calculate_k`).
    - `method="ilp"` without simplices: the edgelist ILP (`calculate_m`).
    - `method="gc"`: PUMA (`puma`).
- **`kamui/utils.py`** builds edges and elementary cycles ("simplices") for 2-D and 3-D grids, including cyclical axes, and holds `prepare_weights`. Cycles must traverse each shared edge in opposite directions; `tests/test_utils.py` checks this orientation invariant.
- **`kamui/core.py`** holds the solvers.
  - `calculate_k` builds the cycle–edge matrix. An entry is +1 if the cycle walks the edge in its stored `(u, v)` direction, −1 otherwise. It splits `k = x⁺ − x⁻` into non-negative variables and solves for them.
  - `calculate_m` solves for per-vertex offsets from an integer difference array.
  - Both go through `_solve_integer_program`. It solves the LP relaxation with HiGHS (`scipy.optimize.linprog`) and re-solves with `integrality=1` only if the LP optimum is fractional; the `ilp_fallback` flag records when that happened.
  - `integrate` recovers the phase. `unwrap_arbitrary` passes it each edge in both directions; it accumulates the corrected differences along a BFS spanning tree, with pointer jumping. Accumulating in DFS visit order was a past bug (#22).
  - `puma` needs the optional `PyMaxflow`. The `extra` extra installs it, and it is always present in the test environments.

Cross-cutting conventions:

- **Solver reports:** every solver returns a `scipy.optimize.OptimizeResult` when `return_info=True`, with fields `fun`, `success`, `status`, `message` and `ilp_fallback`. `unwrap_arbitrary` always requests it internally. A failed solve returns `None` as the result, together with the report.
- **Input validation:** invalid inputs raise `ValueError` or `TypeError` rather than asserting.

## Testing and docs constraints

- **Coverage:** it must stay at 100%. Mark genuinely unreachable lines with `# pragma: no cover`, and give the reason.
- **README examples:** `tests/test_readme.py` executes every python block in `README.md`, in order and in one shared namespace. README examples must run as written and must not reuse variable names in ways that break later blocks.
- **Docstrings:** public functions need numpydoc docstrings that pass `pixi run docstrings`. Ruff's `D` rules use the numpy convention, and tests are exempt.
- **Math on GitHub:** in the README, PRs and issues, write inline math as $`...`$ and display math as `math` fenced blocks. GitHub's renderer rejects `\operatorname`; use `\text{round}` and the like instead.

## Packaging and releases

- **Branches:** `dev` is the only long-lived branch. PRs target it, and releases ship from it.
- **Version:** the version comes from git tags through hatch-vcs, and no file declares it. Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which runs lint, tests, build and smoke, then publishes to PyPI through trusted publishing. Smoke fails if the wheel's version doesn't match the tag.
- **conda recipe:** `conda-recipe/recipe.yaml` (rattler-build v1 format) carries the only hand-written version.
- **SciPy pin:** the `py310` environment pins `scipy<1.15`. conda-forge's SciPy 1.15.0–1.15.2 builds run HiGHS hundreds of times slower, which makes the README test effectively hang.
