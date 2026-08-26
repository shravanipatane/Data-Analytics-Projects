# Week 4 — Predictive Modeling & Optimization in Logistics Systems
### Two-Wheeler Last-Mile Delivery Logistics

## Overview
This project builds a predictive model to forecast **delivery time (minutes)**
for a two-wheeler (scooter/motorcycle) last-mile logistics operation, and uses
the model's insights to propose two concrete **optimization strategies**:
zone-based rider allocation and stop-batching / route consolidation.

Because no proprietary company data was available, a **realistic synthetic
dataset (5,000 orders)** was simulated with statistically grounded
relationships (distance, traffic, weather, number of stops, rider experience,
etc. driving delivery time), mirroring what a real order-management / GPS
tracking export would look like.

## Problem Statement
> Given order-level and contextual features known at the moment of dispatch,
> predict how long a two-wheeler delivery will take (in minutes), and use
> that prediction to optimize rider allocation and route/order batching.

## Project Structure
```
Week-4-Predictive-Modeling-Optimization/
├── data/                 Raw, cleaned, train/test, prediction & optimization CSVs
├── scripts/              01–06 Python pipeline scripts (run in order)
├── models/                Trained model .pkl files
├── visualizations/         PNG charts generated during evaluation/tuning/optimization
├── results/               Performance / CV / feature-importance / optimization CSVs
└── report/                Final Word (.docx) and PDF report
```

## Pipeline (run in order)
| Step | Script | Purpose |
|---|---|---|
| 1 | `01_problem_definition_data_simulation.py` | Defines the problem, simulates raw dataset |
| 2 | `02_data_preparation.py` | Cleans data, imputes missing values, encodes categoricals, splits train/test |
| 3 | `03_model_training.py` | Trains Linear Regression, Decision Tree, Random Forest |
| 4 | `04_model_evaluation.py` | Computes RMSE/MAE/R², generates evaluation visualizations |
| 5 | `05_hyperparameter_tuning.py` | GridSearchCV + 5-fold cross-validation, saves tuned models |
| 6 | `06_logistics_optimization.py` | Zone-based rider allocation & route-batching optimization |

Run with:
```bash
cd scripts
python3 01_problem_definition_data_simulation.py
python3 02_data_preparation.py
python3 03_model_training.py
python3 04_model_evaluation.py
python3 05_hyperparameter_tuning.py
python3 06_logistics_optimization.py
```

## Key Results
- **Best model:** Linear Regression — Test RMSE ≈ 3.46 min, MAE ≈ 2.77 min, R² ≈ 0.91
- **Top predictive features:** `distance_km`, `traffic_level`, `num_stops`, `is_peak_hour`, `weather`
- **Optimization impact:** Batching up to 3 nearby orders per trip is estimated to cut
  per-order delivery time by ~61% and fuel cost by roughly ₹50,000 across the
  5,000-order sample, while zone-level rider allocation flags high-traffic
  Urban/Suburban Evening & Afternoon slots as highest SLA-breach risk needing
  additional riders.

## Report
See `report/Week4_Predictive_Modeling_Optimization_Report.docx` (and the
`.pdf` export) for the full write-up: problem definition, methodology, code
walkthroughs, evaluation, and optimization recommendations.

