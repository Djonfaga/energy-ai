# Next steps · Phase 2 plan: AI models for energy prediction and cost

*Prepared 26 October 2026 from a review of IEA, IRENA and US DOE publications, current benchmarks and open tools.
Sources are listed at the end. This is the starting point for the next build sessions.*

---

## 1. Goal

Build AI models that **predict** (solar, wind, load, prices) and turn those predictions into **costs and decisions**
(system size, dispatch, LCOE, revenue), validated on data the models have never seen, and published openly.

Two projects, sharing one forecasting engine and one cost engine:

| | Project B · Forecast-to-value | Project A · Mali mini-grid AI planner |
|---|---|---|
| Question | How accurate, and how *valuable*, can a renewable/price forecast be? | What should a solar-battery-diesel mini-grid in a Malian village look like, and what does its electricity cost? |
| Data | rich, open, with ground truth (GB, Germany, US) | scarce: solar resource is good, demand data almost absent |
| AI | probabilistic forecasting, foundation models, value-oriented evaluation | demand estimation, forecasting for dispatch, optimisation under uncertainty |
| Cost side | trading revenue, battery arbitrage value | NPC, LCOE, tariff gap, subsidy need |
| Benchmark | the HEFTCom2024 winners and published baselines | ESMAP / Africa Minigrids benchmarks, Malian tariffs |

**Recommended order: B first (about 3 weeks), then A (about 5–6 weeks).** B has real ground truth and a public
leaderboard, so it is where we learn and prove the forecasting and valuation craft. A then reuses that engine in a
context where data is scarce and the impact is largest. *(If you prefer to start with Mali, begin A at step A1;
the forecasting module is only needed in A5.)*

## 2. Principles taken from the research

1. **Value over accuracy.** In HEFTCom2024 (3.6 GW of GB wind and solar), a perfect forecast would have earned
   £92.0m, only £3.2m more than the winner, while perfect *decisions* would have earned £105.2m. Measure every
   forecast by the money or reliability it buys, not only by its error.
2. **Strong, simple baselines first.** Gradient-boosted trees, separate wind/solar models and careful feature work
   won HEFTCom2024; the first notebook series showed the same pattern.
3. **Unseen data only.** Chronological splits and rolling-origin backtests; thresholds and hyper-parameters tuned on
   a validation period, never on the test period.
4. **Probabilistic outputs.** Quantiles (10…90 %) scored with pinball loss / CRPS; calibration checked.
5. **The real bottlenecks are data, skills and trust** (IEA, IRENA). Clean data pipelines, documentation and
   reproducibility are part of the product, not extras.

---

## 3. Project B · Forecast-to-value engine

### B0 · Setup (session 1)
- New repository and environment (see section 5); API keys (section 6).
- Download HEFTCom2024 data from IEEE DataPort (DOI 10.21227/5hn0-8091): power production (Hornsea 1 offshore wind +
  East-England solar), weather forecasts, market prices, participants' submissions.
- Re-create the competition's scoring: pinball loss over quantiles 10…90 % and the trading revenue rule.

### B1 · Baselines (week 1)
- Climatology quantiles, persistence, and a simple power-curve model (wind speed → power; irradiance → PV).
- Rolling-origin evaluation harness reused by every later step.

### B2 · Gradient-boosting quantile models (week 1–2)
- LightGBM / CatBoost quantile regression with NWP features (wind speed at hub height, direction, irradiance,
  temperature, cloud cover, solar geometry, calendar).
- Separate wind and solar models, then combine (as 9 of the top 10 teams did); stacking of sister models trained on
  different weather sources.
- Conformal or quantile-regression post-calibration; reliability diagrams.

### B3 · Foundation models (week 2)
- **Chronos-2** (Amazon, 120 M parameters, Apache-2.0, native covariates): zero-shot, then with weather covariates,
  then fine-tuned. Compare with TimesFM 2.5. In the 2026 load benchmarks on US grids, Chronos-2 was the strongest
  pretrained model.
- Question to answer honestly: does the foundation model beat tuned gradient boosting on this task, and when?

### B4 · From forecast to value (week 2–3)
- Trading strategy: choose the bid from the quantile forecast and the price spread (newsvendor logic); revenue on the
  test period vs the published winners and vs a perfect forecast.
- Show the accuracy–value curve: which forecast improvements actually earn money.

### B5 · Prices and storage (week 3, optional extension)
- Day-ahead price forecasting with the epftoolbox benchmark (EPEX-BE/FR/DE, Nord Pool, PJM): LEAR baseline
  (re-implemented with scikit-learn if the toolbox's dependencies are a problem), then gradient boosting / Chronos-2.
- Value: battery arbitrage revenue (LP from notebook 09/11) with forecast prices vs perfect prices.

**Success criteria for B:** pinball loss and revenue reported on the unseen period next to the HEFTCom2024 leaders;
an explicit accuracy-versus-value analysis; everything reproducible from scripts.

---

## 4. Project A · Mali mini-grid AI planner

### Context (from the research)
- About 250 mini-grids, run by around 68 private operators and EDM, most of them diesel; AMADER manages the rural
  ones, and World Bank, AFD and IsDB programmes are hybridising them with solar PV and batteries.
- The Rural Electrification Fund can subsidise up to 80 % of capex (grants up to USD 500 k).
- Diesel: about 725 XOF per litre (Dec 2025 – Feb 2026). EDM residential tariff about 131 XOF/kWh (June 2025), below
  EDM's cost of supply; historical mini-grid tariffs around 250–335 XOF/kWh.
- ESMAP benchmark: best-in-class solar-hybrid mini-grid LCOE about USD 0.38/kWh (2021), target 0.20 by 2030; component
  benchmarks around USD 596/kWp PV, 297/kWh Li-ion battery, 265/kW battery inverter. The 2024 Africa Minigrids
  benchmark reports a four-year average capex of USD 6 824/kWp.
- *Several of these figures are dated (2013–2022); the model must treat them as uncertain ranges, not constants.*

### Pipeline

```
village location ─► solar & weather (PVGIS, NASA POWER, ERA5, Global Solar Atlas)
       │
       ├─► demand estimate (RAMP archetypes  +  ML from buildings / night lights / population)
       │
       ├─► sizing & dispatch optimisation (PV, battery, inverter, diesel)  ──►  NPC, LCOE, renewable share, CO₂
       │
       ├─► uncertainty (demand scenarios, diesel price, cost ranges)  ──►  robust design, sensitivity
       │
       └─► operation: day-ahead solar & load forecast + MPC dispatch  ──►  diesel saved (XOF)
```

### A1 · Sites and solar resource (week 1)
- Choose 3–5 representative villages (for example in Ségou, Sikasso and Koulikoro regions; or existing hybrid sites).
- Hourly irradiance and temperature: PVGIS API (SARAH satellite data covers Africa; confirm the current version),
  NASA POWER as a cross-check, Global Solar Atlas for PVOUT maps; PV output with pvlib (temperature derating, losses).
- Validate the satellite data against the published ground comparison for Agoufou, Mali (RMSE 20–34 W/m²).

### A2 · Demand estimation (week 2–3), the hardest and most valuable part
- **Bottom-up:** RAMP (`pip install rampdemand`, EUPL) with user archetypes: household tiers (Multi-Tier Framework),
  shops, grain mills, health centre, school, water pumping, productive uses. Calibrate appliance lists and usage
  windows on published African mini-grid studies (Uganda: 25 mini-grids, 2 124 households, 15-minute meters;
  Tanzania: 1 378 customers on 14 mini-grids).
- **Top-down ML:** building footprints (Google Open Buildings / Microsoft), population (WorldPop), night lights
  (VIIRS), distance to roads and towns → demand per site, following the OEMaps approach (about 80 % building-level
  accuracy reported by the IEA). Appliance ownership matters more than demographics: the Tanzanian study's best
  model (LASSO) had 37 % median error per site.
- Deliverable: demand scenarios (low / central / high growth) instead of one profile.
- *Look for real data:* ask AMADER, operators, the Africa Minigrids Program or research groups for anonymised meter
  data; even one site would transform validation.

### A3 · Sizing and cost engine (week 3–4)
- Option 1: **MicroGridsPy** (Linopy + HiGHS, EUPL; multi-year, multi-scenario; PV, wind, battery, generator, grid).
- Option 2: our own Pyomo/HiGHS model (we already have the patterns from notebooks 09, 11 and 17), giving full control
  over subsidies, tariffs and outage penalties.
- Plan: run MicroGridsPy as a reference, build our own model, and cross-check one site between the two.
- Outputs: optimal kWp / kWh / kW, NPC, LCOE, renewable fraction, diesel litres, CO₂, unserved energy; cost vs
  tariff gap and the subsidy needed to reach a target tariff.

### A4 · Uncertainty and robustness (week 4–5)
- Two-stage stochastic sizing over RAMP demand scenarios and diesel price paths; sensitivity of LCOE to PV, battery
  and diesel prices (tornado chart); value of demand information (how much a better demand estimate is worth).

### A5 · Operations AI (week 5–6)
- Day-ahead solar forecast (engine from B, driven by ECMWF open data, CC-BY-4.0) and load forecast.
- MPC battery/genset dispatch (from notebook 11) vs the usual rule-based dispatch; diesel saved per year in XOF.

### A6 · Product and report (week 6)
- Streamlit app: pick a village, get the design, cost, tariff gap and uncertainty band.
- Report and public repository (French summary for Malian stakeholders).

**Success criteria for A:** LCOE results inside published ranges and explained; demand scenarios calibrated on real
African data; a sizing tool cross-checked against MicroGridsPy; a clear statement of what each uncertainty does to
the cost.

---

## 5. Technical setup

- New public repository, e.g. `Djonfaga/energy-ai-planner` (name to confirm), same structure as `energy-ai`:
  `src/` package, `notebooks/py/` sources, `tools/` build and figure export, `results/figures/`, report builder.
- New environment (Python 3.12); key packages: `lightgbm`, `catboost`, `chronos-forecasting>=2.0`, `timesfm`
  (optional), `pvlib`, `rampdemand`, `microgridspy`, `linopy`, `highspy`, `pyomo`, `entsoe-py`, `ecmwf-opendata`,
  `cdsapi`, `geopandas`, `rasterio`, `scoringrules` (CRPS), `streamlit`, plus the existing stack.
- Hardware: Chronos-2 (120 M) inference runs on the CPU or the Intel GPU; fine-tuning is small enough for the
  laptop. Remember the XPU `randperm` workaround and to build notebooks one at a time.

## 6. Accounts and decisions needed before session 1

| Item | Why | Where |
|---|---|---|
| IEEE account | download the HEFTCom2024 data | ieee-dataport.org (DOI 10.21227/5hn0-8091) |
| ENTSO-E API token | European load, generation, prices | transparency.entsoe.eu (register, then request API access) |
| EIA API key (free) | US hourly grid data (EIA-930) | eia.gov/opendata |
| Copernicus CDS account | ERA5 reanalysis | cds.climate.copernicus.eu |
| Elexon API key (if required) | GB market data | elexon.co.uk / Insights Solution |
| Decide: B first or A first | sets the first sprint | see section 1 |
| Optional: contacts in Mali | real mini-grid meter data | AMADER, operators, Africa Minigrids Program |

## 7. Risks

| Risk | Mitigation |
|---|---|
| No open Malian mini-grid demand data | scenario-based demand calibrated on Uganda/Tanzania studies; seek partner data; report value of better data |
| Cost and tariff data are dated | treat as ranges; sensitivity analysis; update when AMADER/CREE publish |
| HEFTCom data licence or access terms | read the DataPort terms first; fall back to ENTSO-E + ERA5 for Germany |
| Some API details unconfirmed (Elexon key, ECMWF 100 m wind in the free subset, PVGIS data version) | verify in session 1 before building on them |
| Foundation models slow on the laptop | zero-shot first, small fine-tunes, cache predictions |

## 8. Session 1 checklist (next time)

1. Confirm the order (B → A recommended) and the repository name.
2. Create the repository, environment and folder structure; copy the build and figure tools from `energy-ai`.
3. Verify data access: HEFTCom2024 download and licence, PVGIS API for one Malian coordinate, Open-Meteo / ECMWF open
   data.
4. Build the evaluation harness (rolling origin, pinball loss, CRPS, revenue) and the first baseline notebook.
5. Commit and push.

---

## Sources

- IEA, *Energy and AI* (2025), executive summary: https://www.iea.org/reports/energy-and-ai/executive-summary
- IRENA, *Digitalisation and AI for power system transformation: Perspectives for the G7* (2025):
  https://www.irena.org/Publications/2025/Oct/Digitalisation-and-AI-for-power-system-transformation-Perspectives-for-the-G7
  and coverage: https://bioenergyinternational.com/irena-unveils-energy-ai-and-digitalisation-report-at-g7/
- US DOE, *AI for Energy* (2024): https://www.energy.gov/node/4840717 ; Genesis Mission:
  https://www.powermag.com/does-genesis-mission-enlists-ai-to-double-u-s-research-productivity-in-a-decade/
- HEFTCom2024 organisers' paper: https://arxiv.org/abs/2507.01579 ; data: https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition ;
  winning model (stacked CatBoost): https://www.sciencedirect.com/science/article/pii/S0169207026000269 ;
  GEB team code: https://github.com/BigdogManLuo/HEFTcom24
- epftoolbox and benchmark (Lago et al., 2021): https://pure.tudelft.nl/ws/portalfiles/portal/90920037/1_s2.0_S0306261921004529_main.pdf
- Chronos-2: https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting ;
  load-forecasting benchmarks: https://arxiv.org/html/2602.10848 and https://alphaxiv.org/abs/2610.07232v1
- ECMWF open data (CC-BY-4.0) and client: https://github.com/ecmwf/ecmwf-opendata ; AIFS for solar:
  https://www.pv-magazine.com/2025/03/21/accuracy-analysis-ecmwfs-ai-model-for-solar-forecasting-performs-well/
- MicroGridsPy: https://github.com/MicroGridsPy/MicroGridsPy ; RAMP: https://github.com/RAMP-project/RAMP
- Mali mini-grids: https://energypedia.info/wiki/Mali_Energy_Situation ;
  https://documents1.worldbank.org/curated/en/397151627676502492/pdf/Concept-Project-Information-Document-PID-Mali-Electricity-System-Reinforcement-and-Access-Expansion-Project-P176633.pdf ;
  https://www.arteliagroup.com/project/hybridation-and-mini-grids-in-mali/ ;
  https://rise.esmap.org/data/files/library/mali/Electricity%20Access/Mali_Evaluation%20of%20rural%20electrification%20concessions.pdf
- Prices: https://www.globalpetrolprices.com/Mali/diesel_prices/ ; https://www.globalpetrolprices.com/Mali/electricity_prices/
- Mini-grid costs: ESMAP *Mini Grids for Half a Billion People* (2022):
  https://www.villanova.edu/content/dam/villanova/engineering/energy-workshop/ESMAP%20Mini%20Grids%20-%20Full%20Report%20Sept.%202022.pdf ;
  *Benchmarking Africa's Minigrids* (2024): https://www.africamda.org/wp-content/uploads/2025/04/Benchmarking-Africas-Minigrids-Report-2024-Online-version.pdf ;
  IRENA *Renewable Power Generation Costs in 2025*: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2026/Jul/IRENA_TEC_RPGC_in_2025_2026.pdf
- Demand data and estimation: Uganda mini-grid load profiles: https://www.sciencedirect.com/science/article/pii/S0301421523005542 ;
  Tanzania: https://mdpi.com/1996-1073/14/14/4207/htm ; IEA on geospatial demand mapping:
  https://www.iea.org/commentaries/africas-electricity-access-planners-turn-to-geospatial-mapping ;
  remote sensing of mini-grids: https://iopscience.iop.org/article/10.1088/2634-4505/ae636d ;
  AI for mini-grids survey: https://arxiv.org/pdf/2407.15865
- Solar resource: Global Solar Atlas maps for Mali (World Bank); SARAH validation in West Africa:
  https://acp.copernicus.org/articles/20/12871/2020/acp-20-12871-2020.html
- Market data: https://pypi.org/project/entsoe-py/ ; https://www.eia.gov/electricity/930-content/about.html ;
  https://github.com/OSUKED/ElexonDataPortal
