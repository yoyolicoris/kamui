import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_examples_run():
    # The Usage examples build on each other, so run them in order in one
    # namespace; a failure names the example it came from.
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    assert blocks
    namespace = {}
    for number, block in enumerate(blocks, start=1):
        exec(compile(block, f"README.md example {number}", "exec"), namespace)
