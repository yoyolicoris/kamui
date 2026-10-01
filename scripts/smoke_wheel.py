"""Install the freshly built wheel into a throwaway venv and check it imports.

``python -m build`` can succeed while producing a wheel that is unusable, and
the version is wired through three hops (git tag, ``hatch-vcs``, then
``importlib.metadata`` at import time), none of which the build itself
verifies. A shallow checkout, for instance, builds cleanly and yields a
placeholder version that would be published for real. Run after ``build``.
"""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path

DIST = Path("dist")


def main() -> int:
    """Install the newest wheel in ``dist/`` and verify its version wiring."""
    wheels = sorted(DIST.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"no wheel in {DIST}/; run `pixi run build` first", file=sys.stderr)
        return 1
    wheel = wheels[-1]
    # kamui-<version>-py3-none-any.whl
    expected = wheel.name.split("-")[1]

    with tempfile.TemporaryDirectory() as tmp:
        venv.create(tmp, with_pip=True)
        win = sys.platform == "win32"
        py = Path(tmp) / ("Scripts" if win else "bin") / ("python.exe" if win else "python")
        subprocess.run([py, "-m", "pip", "install", "--quiet", str(wheel)], check=True)
        got = subprocess.run(
            [py, "-c", "import kamui; print(kamui.__version__)"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if got != expected:
        print(
            f"{wheel.name} declares version {expected} but kamui.__version__ is {got}",
            file=sys.stderr,
        )
        return 1
    if got.startswith("0.0.0"):
        print(
            f"version resolved to {got}: the checkout has no tags or history "
            "(is fetch-depth: 0 set?)",
            file=sys.stderr,
        )
        return 1
    print(f"{wheel.name} installs, imports, and reports kamui {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
