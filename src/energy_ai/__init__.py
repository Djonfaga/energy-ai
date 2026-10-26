"""Energy-system AI models: shared helpers (data loading, device selection, plotting style)."""
from .device import get_device
from .data import opsd, weather, uci_customers, simbench_grid, gb_dataset, clean_load, solar_elevation, day_ahead_price, home_profiles, set_seed, ROOT, RAW, PROCESSED, FIGURES
from .gridsim import snapshots, fast_pf, ev_charging_probability
from .gridplot import plot_grid, bus_xy

__all__ = ["get_device", "opsd", "weather", "uci_customers", "simbench_grid", "set_seed", "ROOT", "RAW", "PROCESSED", "FIGURES",
           "snapshots", "fast_pf", "ev_charging_probability", "gb_dataset", "clean_load", "solar_elevation", "day_ahead_price", "home_profiles", "plot_grid", "bus_xy", "style"]


def style():
    """Consistent, readable plot style for all notebooks."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110, "font.size": 10})
