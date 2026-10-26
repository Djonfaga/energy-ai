"""Generate labelled grid data by running many power flows on SimBench grids (cached in data/processed/).

Each snapshot = one time step of the year: household demand, PV output -> voltages, currents, loadings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp

from .data import PROCESSED, simbench_grid


def fast_pf(net) -> bool:
    """Power flow with options that skip repeated checks (≈2x faster); returns convergence flag."""
    try:
        pp.runpp(net, check_connectivity=False, init="results", lightsim2grid=True)
    except Exception:
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            return False
    return bool(net.converged)


def ev_charging_probability(index: pd.DatetimeIndex) -> np.ndarray:
    """Share of home-charged EVs that are charging at each time step (typical uncontrolled home charging:
    low during the day, a peak around 19:30 when people come home, a tail into the night)."""
    h = index.hour.values + index.minute.values / 60
    p = 0.03 + 0.30 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2) + 0.08 * np.exp(-0.5 * ((h - 23.5) / 1.5) ** 2)
    return np.where(index.dayofweek.values >= 5, 0.7 * p + 0.03, p)


def snapshots(code: str = "1-LV-semiurb4--0-sw", n: int | None = 4000, seed: int = 0,
              load_scale: float = 1.0, pv_scale: float = 1.0, ev_share: float = 0.0, ev_kw: float = 11.0,
              use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Run power flows for n random time steps of the SimBench year (n=None -> every 15-min step).

    ev_share > 0 adds uncontrolled EV home charging (ev_kw per car) to that share of the households (H0 profiles);
    each car charges at a given time step with the probability from `ev_charging_probability`.

    Returns DataFrames indexed by timestamp:
      p_load [MW] per load (household/business demand, EV excluded), q_load [Mvar], p_ev [MW] EV charging per load,
      p_pv [MW] per PV system, vm [p.u.] per bus, line_loading [%] and line_p [MW] per line, trafo_loading [%].
    """
    tag = f"snap_{code}_{n}_{seed}_{load_scale}_{pv_scale}" + (f"_ev{ev_share}_{ev_kw}" if ev_share else "")
    path = PROCESSED / f"{tag.replace('.', 'p')}.parquet"
    keys = ("p_load", "q_load", "p_ev", "p_pv", "vm", "line_loading", "line_p", "trafo_loading")
    if use_cache and path.exists():
        big = pd.read_parquet(path)
        return {k: big[k] for k in keys if k in big.columns.get_level_values(0)}

    net, prof = simbench_grid(code)
    # SimBench "future" grids contain PV-battery storage whose profiles only ever discharge (not energy-consistent);
    # left at their static values they inject a constant ~0.2 MW. Take them out of service.
    if len(net.storage):
        net.storage["in_service"] = False
    idx = prof[("load", "p_mw")].index
    if n is not None and n < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=n, replace=False))
    P, Q = prof[("load", "p_mw")].loc[idx] * load_scale, prof[("load", "q_mvar")].loc[idx] * load_scale
    PV = prof[("sgen", "p_mw")].loc[idx] * pv_scale
    EV = pd.DataFrame(0.0, index=idx, columns=P.columns)
    if ev_share:
        homes = net.load.index[net.load.profile.astype(str).str.startswith("H0")]
        owners = np.random.default_rng(42).choice(homes, size=int(round(ev_share * len(homes))), replace=False)
        charging = np.random.default_rng(seed + 1).random((len(idx), len(owners))) < ev_charging_probability(pd.DatetimeIndex(idx))[:, None]
        EV.loc[:, owners] = charging * ev_kw / 1000
    out = {k: [] for k in ("vm", "line_loading", "line_p", "trafo_loading")}
    ok = []
    for t in idx:
        net.load["p_mw"], net.load["q_mvar"] = P.loc[t].values + EV.loc[t].values, Q.loc[t].values
        if len(net.sgen):
            net.sgen["p_mw"] = PV.loc[t].values
        conv = fast_pf(net)
        ok.append(conv)
        out["vm"].append(net.res_bus.vm_pu.values.copy())
        out["line_loading"].append(net.res_line.loading_percent.values.copy())
        out["line_p"].append(net.res_line.p_from_mw.values.copy())
        out["trafo_loading"].append(net.res_trafo.loading_percent.values.copy())
    ok = np.array(ok)
    res = {
        "p_load": P[ok], "q_load": Q[ok], "p_ev": EV[ok], "p_pv": PV[ok],
        "vm": pd.DataFrame(np.array(out["vm"])[ok], index=idx[ok], columns=[f"bus{b}" for b in net.bus.index]),
        "line_loading": pd.DataFrame(np.array(out["line_loading"])[ok], index=idx[ok], columns=[f"line{l}" for l in net.line.index]),
        "line_p": pd.DataFrame(np.array(out["line_p"])[ok], index=idx[ok], columns=[f"line{l}" for l in net.line.index]),
        "trafo_loading": pd.DataFrame(np.array(out["trafo_loading"])[ok], index=idx[ok], columns=[f"trafo{t}" for t in net.trafo.index]),
    }
    res["p_load"].columns = [f"load{c}" for c in range(res["p_load"].shape[1])]
    res["q_load"].columns = [f"load{c}" for c in range(res["q_load"].shape[1])]
    res["p_ev"].columns = [f"load{c}" for c in range(res["p_ev"].shape[1])]
    res["p_pv"].columns = [f"pv{c}" for c in range(res["p_pv"].shape[1])]
    if use_cache:
        pd.concat(res, axis=1).to_parquet(path)
    return res
