"""Export every figure embedded in the executed notebooks to results/figures/<notebook>/fig_NN.png and write an
index page (results/figures/README.md) that shows each figure under the notebook section it belongs to.

Usage:  python tools/export_figures.py            (all notebooks)
        python tools/export_figures.py 08_cable_overload_prediction
"""
import base64, json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR, FIG_DIR = ROOT / "notebooks", ROOT / "results" / "figures"


def export(nb_path: Path) -> list[tuple[str, str]]:
    """Write the notebook's PNG outputs; return [(section heading, relative png path)]."""
    nb = json.loads(nb_path.read_text())
    out_dir = FIG_DIR / nb_path.stem
    shutil.rmtree(out_dir, ignore_errors=True)
    section, found = nb_path.stem, []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heads = re.findall(r"^#{1,3}\s+(.+)$", src, flags=re.M)
            if heads:
                section = heads[-1].strip()
            continue
        for o in cell.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png:
                out_dir.mkdir(parents=True, exist_ok=True)
                name = f"fig_{len(found) + 1:02d}.png"
                (out_dir / name).write_bytes(base64.b64decode(png))
                found.append((section, f"{nb_path.stem}/{name}"))
    return found


def main(names):
    paths = [NB_DIR / f"{n}.ipynb" for n in names] if names else sorted(NB_DIR.glob("[0-9]*.ipynb"))
    for p in paths:
        print(f"{p.stem}: {len(export(p))} figures")
    # the index always covers every notebook (re-reading is cheap and keeps it consistent)
    lines = ["# Figures", "", "Every figure of the executed notebooks, exported by `tools/export_figures.py`.",
             "Open the notebook for the code and the explanation around each figure.", ""]
    for p in sorted(NB_DIR.glob("[0-9]*.ipynb")):
        figs = export(p)
        if not figs:
            continue
        lines += [f"## [{p.stem}](../../notebooks/{p.name})", ""]
        last = None
        for section, rel in figs:
            if section != last:
                lines += [f"**{section}**", ""]
                last = section
            lines += [f"![{section}]({rel})", ""]
    (FIG_DIR / "README.md").write_text("\n".join(lines))
    print(f"index written: {FIG_DIR / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
