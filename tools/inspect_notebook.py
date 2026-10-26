"""Print the text outputs and errors of an executed notebook; optionally save its figures as PNG files.

Usage:  python tools/inspect_notebook.py notebooks/08_cable_overload_prediction.ipynb [figure_dir]
"""
import base64, json, os, sys

path = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else None
if outdir:
    os.makedirs(outdir, exist_ok=True)
nb = json.load(open(path))
n_img = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            print(f"[{i}]", "".join(o["text"])[:1200].rstrip())
        elif o["output_type"] in ("execute_result", "display_data"):
            d = o["data"]
            if "image/png" in d:
                n_img += 1
                if outdir:
                    name = f"img_{os.path.basename(path)[:2]}_{n_img}.png"
                    with open(os.path.join(outdir, name), "wb") as fh:
                        fh.write(base64.b64decode(d["image/png"]))
            elif "text/plain" in d:
                print(f"[{i}]", "".join(d["text/plain"])[:800])
        elif o["output_type"] == "error":
            print(f"[{i}] ERROR", o["ename"], o["evalue"][:500])
print("images:", n_img)
