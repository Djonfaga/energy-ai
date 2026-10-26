# Energy AI: from power flow to reinforcement learning

A hands-on path through **AI for intelligent and integrated energy systems**, in 18 executed Jupyter notebooks (00–17). It
starts with a power flow on a four-house feeder and ends with physics-informed neural networks, learned control
policies, cyber attacks on state estimation and multi-energy hubs.

Every notebook uses **open real-world data** (European load, generation, prices and weather) or **standard benchmark
grids** (SimBench, IEEE 14-bus), checks its machine-learning results against physics simulations, and states plainly
when a simple method wins.

The topics follow the syllabus of the TU Delft / edX course *Digitalization of Intelligent and Integrated Energy
Systems* (DelftX IIES02x). This repository is an independent study project: it is not affiliated with TU Delft or
edX and contains no course material.

## Learning path

| # | Notebook | Level | Methods | Course module | Headline result |
|---|---|---|---|---|---|
| 00 | [check_device](notebooks/00_check_device.ipynb) | setup | PyTorch on CPU / CUDA / Intel XPU | – | 4096² matrix product 9.5× faster on an Intel Iris Xe iGPU |
| 01 | [power_flow_basics](notebooks/01_power_flow_basics.ipynb) | 1 · simulation | AC power flow (pandapower), SimBench, time series | M2 | a 4-house feeder hits its voltage limit at ≈ 80 kW, its thermal limit at ≈ 89 kW; PV back-feed causes the highest loading |
| 02 | [numerical_simulation_methods](notebooks/02_numerical_simulation_methods.ipynb) | 1 · simulation | Euler / RK4 / implicit ODE solvers, stiffness, Newton–Raphson vs Gauss–Seidel, AC vs DC, co-simulation | M2 | stiff frequency dynamics: RK45 needs 3 025 steps, BDF 169 |
| 03 | [load_profile_clustering_kmeans](notebooks/03_load_profile_clustering_kmeans.ipynb) | 2 · descriptive | k-means, silhouette, profile features | M4 | shops & services use 33 % of the energy but cause 50 % of the system peak |
| 04 | [anomaly_detection_autoencoder](notebooks/04_anomaly_detection_autoencoder.ipynb) | 2 · descriptive | autoencoder (GPU), PCA, isolation forest | M4 / M5 | autoencoder flags 87 % of false-data injections at 1.7 % false alarms (AUC 0.94 vs 0.74) |
| 05 | [forecasting_regression](notebooks/05_forecasting_regression.ipynb) | 3 · predictive | linear / polynomial regression, data cleaning, holidays | M4 | load MAPE 6.1 % → 5.0 %, solar nRMSE 9.7 % → 6.2 % |
| 06 | [forecasting_ensembles_neural](notebooks/06_forecasting_ensembles_neural.ipynb) | 3 · predictive | trees, random forest, boosting, SVR, MLP, stacking, meta-NN, quantile regression | M4 | day-ahead load MAPE 3.45 %; meta-network ensemble as used by National Grid ESO |
| 07 | [dynamic_security_assessment](notebooks/07_dynamic_security_assessment.ipynb) | 3 · predictive | frequency dynamics, decision trees, cost-sensitive classification | M4 | the tree rediscovers the RoCoF limit E_kin ≥ 50·P_loss; zero missed alarms at threshold 0.1 |
| 08 | [cable_overload_prediction](notebooks/08_cable_overload_prediction.ipynb) | 3 · **capstone** | logistic regression, trees, boosting, MLP, permutation importance | M4 final exercise | smart meters predict cable overloads (F1 0.92) without topology; the model finds the meters behind the cable |
| 09 | [optimal_scheduling_milp](notebooks/09_optimal_scheduling_milp.ipynb) | 4 · prescriptive | LP dispatch, MILP unit commitment, storage, Pareto front (Pyomo + HiGHS) | M3 | a battery saves 6 % but raises CO₂; abatement cost rises from 21 to 55 €/t |
| 10 | [surrogate_power_flow](notebooks/10_surrogate_power_flow.ipynb) | 4 · simulation + ML | neural surrogate (GPU), linear surrogate, extrapolation test | M4 | a full year of power flows in 0.3 s (×3 000) with overload hours within ±2 h, but errors up to 54 V once PV ×4 reverses the flows |
| 11 | [mpc_home_energy](notebooks/11_mpc_home_energy.ipynb) | 4 · decision under uncertainty | model predictive control, stochastic and robust MPC (LP) | M3 | two-layer MPC beats a battery rule by 11 %; naive stochastic and robust MPC do not help |
| 12 | [imitation_vs_reinforcement_learning](notebooks/12_imitation_vs_reinforcement_learning.ipynb) | 5 · learning control | behaviour cloning, DAgger, Q-learning, DQN, safety shield (Gymnasium) | M4 | imitation recovers 98 % of the MPC expert's saving; RL after 14 simulated years only 63–69 %, below a one-line rule |
| 13 | [physics_informed_power_flow](notebooks/13_physics_informed_power_flow.ipynb) | 5 · physics + ML | PINN with ΔP/ΔQ and Z-bus voltage residuals, loss conditioning | M4 | with zero labels the PINN matches a network trained on 1 000 power flows; the textbook ΔP/ΔQ loss (condition number ≈ 18 600) fails |
| 14 | [cybersecurity_state_estimation](notebooks/14_cybersecurity_state_estimation.ipynb) | 5 · security | WLS state estimation, χ² bad-data detection, stealthy FDI, ML detectors | M5 | χ² test is blind to stealthy attacks; boosting catches 83 % at 0.8 % false alarms |
| 15 | [congestion_management_flexibility](notebooks/15_congestion_management_flexibility.ipynb) | 6 · integration | EV smart charging LP, radial PTDF, AC validation | M6 | grid-blind smart charging overloads the cable to 337 %; grid-aware charging fixes all limits for +1 % cost |
| 16 | [digital_twin_building](notebooks/16_digital_twin_building.ipynb) | 6 · integration | grey-box calibration, Kalman filter, innovation-based fault detection | M1 | twin + Kalman filter tracks indoor temperature to 0.11 K and spots a COP fault within a day |
| 17 | [integrated_energy_hub](notebooks/17_integrated_energy_hub.ipynb) | 6 · integration | energy hub LP: heat pump, CHP, boiler, heat storage | integrated systems | in a cold, windless week a gas boiler is cleaner than a heat pump; timing matters |

Course modules: M1 digital twins · M2 numerical simulation · M3 decision support · M4 machine learning ·
M5 cybersecurity · M6 grid congestion management.

## Quick start

```bash
git clone <this repository> energy_ai && cd energy_ai
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m ipykernel install --user --name energy-ai-gpu --display-name "Energy AI (Intel GPU)"
.venv/bin/jupyter lab notebooks/
```

**GPU.** `get_device()` picks CUDA, Intel XPU or the CPU automatically. Every notebook also runs on a CPU; only the
neural-network parts get faster on a GPU.
- NVIDIA: the default `torch` wheel includes CUDA.
- Intel integrated / Arc GPUs: `uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/xpu`.
  On some integrated GPUs `torch.randperm` fails on the device; the notebooks create permutations on the CPU and move them.

**Data** is downloaded on first use into `data/raw/` and cached in `data/processed/`. This is about 400 MB, mostly
the OPSD time series and the UCI customer data. The first run of the grid notebooks also simulates a few thousand
power flows (a few minutes); the results are cached.

**Rebuilding notebooks.** The notebooks are generated from plain-Python sources in `notebooks/py/` (jupytext
percent format, easy to diff and review):

```bash
.venv/bin/python tools/build_notebooks.py 08_cable_overload_prediction    # or --all
```

`tools/sync_markdown.py <name>` copies text-only edits into an executed notebook without re-running it (it refuses
if any code cell changed).

## Repository layout

| Path | Content |
|---|---|
| `notebooks/` | executed notebooks, numbered from easy to advanced |
| `notebooks/py/` | their sources (percent-format Python) |
| `src/energy_ai/` | shared code: data loaders, power-flow snapshot generator, device selection, plot style |
| `tools/` | build and inspection scripts |
| `data/`, `models/`, `results/` | created at run time (not versioned) |

## Data sources

All are open and downloaded automatically; please check each provider's licence before redistributing data.

- [Open Power System Data](https://data.open-power-system-data.org/time_series/): hourly load, wind, solar and day-ahead prices (Europe).
- [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api): hourly weather.
- [UCI ElectricityLoadDiagrams 2011–2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014): 370 Portuguese customers.
- [SimBench](https://simbench.de): German benchmark distribution grids with yearly 15-minute profiles (via the `simbench` package).
- IEEE 14-bus test system (via `pandapower.networks`).

## Licence

Code: MIT (see `LICENSE`).
