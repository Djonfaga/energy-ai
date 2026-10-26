"""Draw a pandapower grid from its bus coordinates: cables, transformer, customers and PV systems."""
from __future__ import annotations

import json

import numpy as np


def bus_xy(net) -> np.ndarray:
    """(n_bus, 2) array of bus coordinates, read from the GeoJSON column net.bus.geo."""
    return np.array([json.loads(g)["coordinates"] for g in net.bus.geo])


def plot_grid(net, ax, title: str | None = None, highlight_line=None, highlight_buses=None):
    """Plot the grid on `ax`. Optional: one line drawn thick red, a set of buses drawn orange."""
    xy = bus_xy(net)
    pos = {b: i for i, b in enumerate(net.bus.index)}
    for li, ln in net.line.iterrows():
        a, b = xy[pos[ln.from_bus]], xy[pos[ln.to_bus]]
        hit = li == highlight_line
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red" if hit else "0.55", lw=4 if hit else 1.2, zorder=3 if hit else 1)
    for _, tr in net.trafo.iterrows():
        a, b = xy[pos[tr.hv_bus]], xy[pos[tr.lv_bus]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=2.5, zorder=2)
        ax.scatter(*b, marker="s", s=110, color="k", zorder=5, label="MV/LV transformer")
    load_buses = [pos[b] for b in set(net.load.bus)]
    hl = {pos[b] for b in (highlight_buses or [])}
    normal = [i for i in load_buses if i not in hl]
    ax.scatter(xy[normal, 0], xy[normal, 1], s=22, color="tab:blue", zorder=4, label="customers")
    if hl:
        hb = [i for i in load_buses if i in hl]
        ax.scatter(xy[hb, 0], xy[hb, 1], s=22, color="tab:orange", zorder=4, label="highlighted customers")
    if len(net.sgen):
        pv = [pos[b] for b in set(net.sgen.bus)]
        ax.scatter(xy[pv, 0], xy[pv, 1], s=90, marker="^", facecolor="none", edgecolor="goldenrod", lw=1.8, zorder=6,
                   label="PV systems")
    ax.set(xticks=[], yticks=[], aspect="equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax
