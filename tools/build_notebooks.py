"""Turn the percent-format sources in notebooks/py/ into executed notebooks in notebooks/.

Usage:  python tools/build_notebooks.py 08_cable_overload_prediction [09_...]     (names without .py)
        python tools/build_notebooks.py --all
"""
import sys, time
from pathlib import Path
import jupytext, nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks" / "py", ROOT / "notebooks"
KERNEL = {"display_name": "Energy AI (Intel GPU)", "language": "python", "name": "energy-ai-gpu"}

names = sorted(p.stem for p in SRC.glob("[0-9]*.py")) if sys.argv[1:] == ["--all"] else sys.argv[1:]
for name in names:
    nb = jupytext.read(SRC / f"{name}.py")
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata.pop("jupytext", None)
    t = time.time()
    client = NotebookClient(nb, timeout=1800, kernel_name=KERNEL["name"], resources={"metadata": {"path": str(OUT)}})
    try:
        client.execute()
        status = "OK"
    except Exception as e:                      # keep partial outputs for debugging
        status = f"FAILED: {type(e).__name__}: {str(e)[-1500:]}"
    for c in nb.cells:                          # drop noisy stderr warnings from outputs
        if c.cell_type == "code":
            c.outputs = [o for o in c.outputs if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
    nbformat.write(nb, OUT / f"{name}.ipynb")
    print(f"{name}: {status}  ({time.time() - t:.0f} s)", flush=True)
