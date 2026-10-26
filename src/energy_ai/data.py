"""Open datasets used in the notebooks. Everything is downloaded once and cached in data/raw/.

- Open Power System Data (OPSD): hourly national load, solar and wind generation (Europe).
- Open-Meteo archive: historical hourly weather for any location (no API key).
- UCI ElectricityLoadDiagrams2011-2014: 15-min consumption of 370 Portuguese customers.
- SimBench: German benchmark grids with one year of 15-min load / PV / wind profiles (ships with the package).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
for _p in (RAW, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

OPSD_URL = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        print(f"downloading {url} -> {path.name} (once)")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


# ---------------------------------------------------------------- OPSD
def opsd(country: str = "NL", start: str = "2016-01-01", end: str = "2019-12-31") -> pd.DataFrame:
    """Hourly load and renewable generation [MW] for one country, indexed by UTC time."""
    path = _download(OPSD_URL, RAW / "opsd_time_series_60min.csv")
    wanted = {
        f"{country}_load_actual_entsoe_transparency": "load_mw",
        f"{country}_solar_generation_actual": "solar_mw",
        f"{country}_wind_onshore_generation_actual": "wind_onshore_mw",
        f"{country}_wind_offshore_generation_actual": "wind_offshore_mw",
    }
    header = pd.read_csv(path, nrows=0).columns
    cols = ["utc_timestamp"] + [c for c in wanted if c in header]
    df = pd.read_csv(path, usecols=cols, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df = df.rename(columns=wanted).loc[start:end]
    return df


# ---------------------------------------------------------------- Open-Meteo
def weather(lat: float, lon: float, start: str, end: str,
            variables=("temperature_2m", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                       "cloud_cover", "wind_speed_100m", "relative_humidity_2m")) -> pd.DataFrame:
    """Hourly historical weather (UTC) from the Open-Meteo archive API, cached as CSV."""
    key = f"weather_{lat:.2f}_{lon:.2f}_{start}_{end}_{len(variables)}.csv"
    path = RAW / key
    if not path.exists():
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                    "hourly": ",".join(variables), "timezone": "UTC"})
        url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
        print("downloading weather from Open-Meteo (once)")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.load(r)["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time").to_csv(path)
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df


# ---------------------------------------------------------------- UCI customers
def uci_customers(year: int = 2014, resample: str = "1h") -> pd.DataFrame:
    """Consumption of 370 customers [kW], one column per customer (customers without data that year are dropped)."""
    cache = PROCESSED / f"uci_customers_{year}_{resample}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zpath = _download(UCI_URL, RAW / "uci_electricityloaddiagrams.zip")
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".txt") and not n.startswith("__")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.loc[str(year)]
    # values are kW averaged over 15 min; resample to the requested resolution
    df = df.resample(resample).mean()
    df = df.loc[:, (df > 0).mean() > 0.95]          # keep customers active almost all year
    df.to_parquet(cache)
    return df


# ---------------------------------------------------------------- SimBench
def simbench_grid(code: str = "1-LV-rural1--0-sw"):
    """Return (net, profiles) for a SimBench grid; profiles are absolute values with 15-min resolution."""
    import simbench as sb
    net = sb.get_simbench_net(code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    idx = pd.date_range("2016-01-01", periods=len(profiles[("load", "p_mw")]), freq="15min")
    for k in profiles:
        if len(profiles[k]):
            profiles[k].index = idx
    return net, profiles


GB_SITES = {"London": (51.51, -0.13), "Birmingham": (52.49, -1.89), "Bristol": (51.45, -2.59),
            "Manchester": (53.48, -2.24), "Edinburgh": (55.95, -3.19)}


def solar_elevation(index: pd.DatetimeIndex, lat: float = 52.5, lon: float = -2.0) -> np.ndarray:
    """Approximate solar elevation angle [deg] (declination + hour angle formula), index in UTC."""
    doy = index.dayofyear.values
    hour = index.hour.values + index.minute.values / 60
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (hour + lon / 15 - 12))
    phi = np.radians(lat)
    sin_el = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def clean_load(load: pd.Series, band: float = 0.6) -> tuple[pd.Series, pd.Series]:
    """Remove physically implausible values from an hourly demand series; returns (cleaned, flagged mask).

    Rule 1: compare each hour with the median of the same hour 1 and 2 weeks before/after. Real demand never falls below
            60 % of that reference (even Christmas only drops to ~61 %), so values outside [band, 1/band] are errors.
    Rule 2: an isolated one-hour spike (>25 % away from both neighbours, while the neighbours agree) is an error.
    Flagged hours are filled by interpolation (gaps ≤ 12 h) or by the weekly reference.
    """
    ref = pd.concat([load.shift(k) for k in (-336, -168, 168, 336)], axis=1).median(axis=1)
    ratio = load / ref
    bad = (ratio < band) | (ratio > 1 / band)
    cleaned = load.mask(bad).interpolate(limit=12).fillna(ref)
    for _ in range(2):
        nb = (cleaned.shift(1) + cleaned.shift(-1)) / 2
        spike = ((cleaned - nb).abs() > 0.25 * nb) & ((cleaned.shift(1) - cleaned.shift(-1)).abs() < 0.2 * nb)
        bad |= spike
        cleaned = cleaned.mask(spike).interpolate(limit=12).fillna(ref)
    return cleaned, bad


def gb_dataset(start: str = "2016-01-01", end: str = "2019-12-31") -> pd.DataFrame:
    """Hourly Great-Britain dataset: national load / solar / wind [MW] (OPSD) + weather averaged over 5 sites
    (Open-Meteo) + calendar and solar-geometry features. Index in UTC."""
    cache = PROCESSED / f"gb_dataset_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    g = opsd("GB_GBN", start, end)
    ws = [weather(la, lo, start, end) for la, lo in GB_SITES.values()]
    w = sum(ws) / len(ws)
    w.index = w.index.tz_convert("UTC")
    g.index = g.index.tz_convert("UTC") if g.index.tz is not None else g.index.tz_localize("UTC")
    df = g.join(w, how="inner")
    t = df.index
    df["hour"], df["dayofweek"], df["month"], df["dayofyear"] = t.hour, t.dayofweek, t.month, t.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * t.dayofyear / 365.25), np.cos(2 * np.pi * t.dayofyear / 365.25)
    df["solar_elevation"] = solar_elevation(t)
    df["is_weekend"] = (t.dayofweek >= 5).astype(int)
    df = df.interpolate(limit=3).dropna()
    df.to_parquet(cache)
    return df


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def day_ahead_price(zone: str = "DE_LU", start: str = "2019-01-01", end: str = "2019-12-31") -> pd.Series:
    """Hourly day-ahead electricity price [€/MWh] for a bidding zone (OPSD; DE_LU available from Oct 2018)."""
    path = _download(OPSD_URL, RAW / "opsd_time_series_60min.csv")
    col = f"{zone}_price_day_ahead"
    s = pd.read_csv(path, usecols=["utc_timestamp", col], parse_dates=["utc_timestamp"], index_col="utc_timestamp")[col]
    return s.loc[start:end].interpolate(limit=3).rename("price_eur_mwh")


def home_profiles(household: str = "H0-B", annual_kwh: float = 4500, pv_kwp: float = 8.0,
                  fees_eur_kwh: float = 0.25) -> pd.DataFrame:
    """Hourly 2019 data for one smart home (local time): household load and PV [kW] from SimBench profiles
    (rotated so weekdays match 2019), DE-LU day-ahead price [€/MWh] and the dynamic import tariff [€/kWh]."""
    net, prof = simbench_grid("1-LV-semiurb4--0-sw")
    hours = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h")

    def to_2019(series):
        h = series.resample("1h").mean().values          # 2016 profile year starts on a Friday
        return np.roll(h, -96)[:len(hours)]              # start on Tue 5 Jan 2016 = weekday of 1 Jan 2019

    hh = net.load.index[net.load.profile == household][0]
    load = to_2019(prof[("load", "p_mw")][hh]) * 1000
    load *= annual_kwh / load.sum()
    pv = to_2019(prof[("sgen", "p_mw")].iloc[:, 0]) / net.sgen.p_mw.iloc[0] * pv_kwp
    price = day_ahead_price("DE_LU", "2018-12-31", "2020-01-01").tz_convert("Europe/Berlin").tz_localize(None)
    price = price.groupby(level=0).mean().reindex(hours).interpolate().values
    return pd.DataFrame({"load_kw": load, "pv_kw": pv, "price_eur_mwh": price, "buy_eur_kwh": price / 1000 + fees_eur_kwh},
                        index=hours)
