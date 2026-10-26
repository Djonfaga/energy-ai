# Figures

Every figure of the executed notebooks, exported by `tools/export_figures.py`.
Open the notebook for the code and the explanation around each figure.

## [01_power_flow_basics](../../notebooks/01_power_flow_basics.ipynb)

**2. Build a small LV feeder by hand**

![2. Build a small LV feeder by hand](01_power_flow_basics/fig_01.png)

**3. A realistic grid: SimBench rural LV network**

![3. A realistic grid: SimBench rural LV network](01_power_flow_basics/fig_02.png)

**Time-series power flow: one summer day**

![Time-series power flow: one summer day](01_power_flow_basics/fig_03.png)

## [02_numerical_simulation_methods](../../notebooks/02_numerical_simulation_methods.ipynb)

**Fixed-step methods written by hand**

![Fixed-step methods written by hand](02_numerical_simulation_methods/fig_01.png)

**Accuracy vs step size (convergence order)**

![Accuracy vs step size (convergence order)](02_numerical_simulation_methods/fig_02.png)

**3. Coupling energy domains: heat pump + building + grid (co-simulation)**

![3. Coupling energy domains: heat pump + building + grid (co-simulation)](02_numerical_simulation_methods/fig_03.png)

## [03_load_profile_clustering_kmeans](../../notebooks/03_load_profile_clustering_kmeans.ipynb)

**The k-means algorithm**

![The k-means algorithm](03_load_profile_clustering_kmeans/fig_01.png)

**2. How many clusters? Elbow and silhouette**

![2. How many clusters? Elbow and silhouette](03_load_profile_clustering_kmeans/fig_02.png)

![2. How many clusters? Elbow and silhouette](03_load_profile_clustering_kmeans/fig_03.png)

![2. How many clusters? Elbow and silhouette](03_load_profile_clustering_kmeans/fig_04.png)

**3. Seeing the clusters: a 2-D projection**

![3. Seeing the clusters: a 2-D projection](03_load_profile_clustering_kmeans/fig_05.png)

## [04_anomaly_detection_autoencoder](../../notebooks/04_anomaly_detection_autoencoder.ipynb)

**3. Train the autoencoder (normal data only)**

![3. Train the autoencoder (normal data only)](04_anomaly_detection_autoencoder/fig_01.png)

**4. Results**

![4. Results](04_anomaly_detection_autoencoder/fig_02.png)

**Which sensor is wrong? Localisation from per-feature errors**

![Which sensor is wrong? Localisation from per-feature errors](04_anomaly_detection_autoencoder/fig_03.png)

## [05_forecasting_regression](../../notebooks/05_forecasting_regression.ipynb)

**Simple linear regression: one input**

![Simple linear regression: one input](05_forecasting_regression/fig_01.png)

**Multiple linear regression: more physics in the features**

![Multiple linear regression: more physics in the features](05_forecasting_regression/fig_02.png)

**2. Overfitting in one picture: polynomial degree**

![2. Overfitting in one picture: polynomial degree](05_forecasting_regression/fig_03.png)

**First: clean the data**

![First: clean the data](05_forecasting_regression/fig_04.png)

**A linear demand model**

![A linear demand model](05_forecasting_regression/fig_05.png)

![A linear demand model](05_forecasting_regression/fig_06.png)

## [06_forecasting_ensembles_neural](../../notebooks/06_forecasting_ensembles_neural.ipynb)

**2. No single model wins everywhere**

![2. No single model wins everywhere](06_forecasting_ensembles_neural/fig_01.png)

**3. The UK National Grid architecture: a meta neural network over weak learners**

![3. The UK National Grid architecture: a meta neural network over weak learners](06_forecasting_ensembles_neural/fig_02.png)

**4. Uncertainty: prediction intervals with quantile regression**

![4. Uncertainty: prediction intervals with quantile regression](06_forecasting_ensembles_neural/fig_03.png)

**Prediction on the unseen test year**

![Prediction on the unseen test year](06_forecasting_ensembles_neural/fig_04.png)

## [07_dynamic_security_assessment](../../notebooks/07_dynamic_security_assessment.ipynb)

**1. The physics: frequency after losing an infeed**

![1. The physics: frequency after losing an infeed](07_dynamic_security_assessment/fig_01.png)

**3. A decision tree an operator can read**

![3. A decision tree an operator can read](07_dynamic_security_assessment/fig_02.png)

**4. Evaluating the classifier: confusion matrix and the cost of mistakes**

![4. Evaluating the classifier: confusion matrix and the cost of mistakes](07_dynamic_security_assessment/fig_03.png)

**Eliminating missed alarms**

![Eliminating missed alarms](07_dynamic_security_assessment/fig_04.png)

**5. Why bother: speed**

![5. Why bother: speed](07_dynamic_security_assessment/fig_05.png)

## [08_cable_overload_prediction](../../notebooks/08_cable_overload_prediction.ipynb)

**1. When and why does the cable overload?**

![1. When and why does the cable overload?](08_cable_overload_prediction/fig_01.png)

**2. Where is the cable, and who is behind it?**

![2. Where is the cable, and who is behind it?](08_cable_overload_prediction/fig_02.png)

**Predictions on the unseen test period (October–December)**

![Predictions on the unseen test period (October–December)](08_cable_overload_prediction/fig_03.png)

**5. Did the model discover the grid topology?**

![5. Did the model discover the grid topology?](08_cable_overload_prediction/fig_04.png)

**6. How many smart meters do we need?**

![6. How many smart meters do we need?](08_cable_overload_prediction/fig_05.png)

**7. Beyond yes/no: predicting the loading itself**

![7. Beyond yes/no: predicting the loading itself](08_cable_overload_prediction/fig_06.png)

## [09_optimal_scheduling_milp](../../notebooks/09_optimal_scheduling_milp.ipynb)

**3. Economic dispatch (LP) and the merit order**

![3. Economic dispatch (LP) and the merit order](09_optimal_scheduling_milp/fig_01.png)

**4. Unit commitment (MILP): on/off decisions, minimum output, start-up costs**

![4. Unit commitment (MILP): on/off decisions, minimum output, start-up costs](09_optimal_scheduling_milp/fig_02.png)

![4. Unit commitment (MILP): on/off decisions, minimum output, start-up costs](09_optimal_scheduling_milp/fig_03.png)

**5. Adding a battery: 100 MW / 400 MWh**

![5. Adding a battery: 100 MW / 400 MWh](09_optimal_scheduling_milp/fig_04.png)

**6. Cost versus emissions: the Pareto front**

![6. Cost versus emissions: the Pareto front](09_optimal_scheduling_milp/fig_05.png)

**7. Sensitivity analysis: how big should the battery be?**

![7. Sensitivity analysis: how big should the battery be?](09_optimal_scheduling_milp/fig_06.png)

## [10_surrogate_power_flow](../../notebooks/10_surrogate_power_flow.ipynb)

**2. Neural-network surrogate on the GPU**

![2. Neural-network surrogate on the GPU](10_surrogate_power_flow/fig_01.png)

**4. Using the surrogate: a full-year planning study**

![4. Using the surrogate: a full-year planning study](10_surrogate_power_flow/fig_02.png)

**5a. Sample efficiency**

![5a. Sample efficiency](10_surrogate_power_flow/fig_03.png)

## [11_mpc_home_energy](../../notebooks/11_mpc_home_energy.ipynb)

**1. One year of data**

![1. One year of data](11_mpc_home_energy/fig_01.png)

**5. What the controllers actually do**

![5. What the controllers actually do](11_mpc_home_energy/fig_02.png)

**The receding horizon in one picture**

![The receding horizon in one picture](11_mpc_home_energy/fig_03.png)

## [12_imitation_vs_reinforcement_learning](../../notebooks/12_imitation_vs_reinforcement_learning.ipynb)

**The compounding-error problem**

![The compounding-error problem](12_imitation_vs_reinforcement_learning/fig_01.png)

**4b. Deep Q-network (DQN)**

![4b. Deep Q-network (DQN)](12_imitation_vs_reinforcement_learning/fig_02.png)

**5. Results on the test month (September 2019)**

![5. Results on the test month (September 2019)](12_imitation_vs_reinforcement_learning/fig_03.png)

## [13_physics_informed_power_flow](../../notebooks/13_physics_informed_power_flow.ipynb)

**4. Few labels: data-only network vs PINN**

![4. Few labels: data-only network vs PINN](13_physics_informed_power_flow/fig_01.png)

## [14_cybersecurity_state_estimation](../../notebooks/14_cybersecurity_state_estimation.ipynb)

**4. Two attacks**

![4. Two attacks](14_cybersecurity_state_estimation/fig_01.png)

**7. Defence by design: protect a basic measurement set**

![7. Defence by design: protect a basic measurement set](14_cybersecurity_state_estimation/fig_02.png)

## [15_congestion_management_flexibility](../../notebooks/15_congestion_management_flexibility.ipynb)

**4. Verification with AC power flow**

![4. Verification with AC power flow](15_congestion_management_flexibility/fig_01.png)

## [16_digital_twin_building](../../notebooks/16_digital_twin_building.ipynb)

**3. Synchronisation with a Kalman filter**

![3. Synchronisation with a Kalman filter](16_digital_twin_building/fig_01.png)

**4. Monitoring: the twin as a fault detector**

![4. Monitoring: the twin as a fault detector](16_digital_twin_building/fig_02.png)

## [17_integrated_energy_hub](../../notebooks/17_integrated_energy_hub.ipynb)

**2. How the hybrid hub operates (winter week, carbon price 80 €/t)**

![2. How the hybrid hub operates (winter week, carbon price 80 €/t)](17_integrated_energy_hub/fig_01.png)
