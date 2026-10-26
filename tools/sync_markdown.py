"""Copy edited markdown from notebooks/py/<name>.py into the executed notebook without re-running it.

Only allowed when every code cell is unchanged (same sources, same order); otherwise rebuild with build_notebooks.py.
Usage:  python tools/sync_markdown.py 13_physics_informed_power_flow
"""
import sys
from pathlib import Path
import jupytext, nbformat

ROOT = Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    new = jupytext.read(ROOT / "notebooks" / "py" / f"{name}.py")
    old = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    old_code = [c for c in old.cells if c.cell_type == "code"]
    new_code = [c for c in new.cells if c.cell_type == "code"]
    if [c.source for c in old_code] != [c.source for c in new_code]:
        sys.exit(f"{name}: code cells differ -> rebuild the notebook instead")
    it = iter(old_code)
    old.cells = [next(it) if c.cell_type == "code" else nbformat.v4.new_markdown_cell(c.source) for c in new.cells]
    nbformat.write(old, ROOT / "notebooks" / f"{name}.ipynb")
    print(f"{name}: markdown synced")
