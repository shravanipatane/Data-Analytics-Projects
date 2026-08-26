"""
06_logistics_optimization.py

OPTIMIZATION OBJECTIVE
-----------------------
Use the trained predictive model (Linear Regression, selected as the
production model in Step 04 for its best generalisation and stability) as a
"what-if" simulator, then apply two optimization strategies that a two-wheeler
logistics operator can act on immediately:

  1. ZONE-BASED RIDER ALLOCATION (resource allocation optimization)
     For every combination of area_type x traffic_level x time_of_day we
     compute the model-predicted average delivery time and translate it into
     a recommended number of concurrent riders needed to keep the promised
     delivery SLA (<= 35 minutes) — using Little's Law style reasoning
     (riders_needed = expected_orders_per_hour * avg_service_time_hours).

  2. STOP-BATCHING / ROUTE CONSOLIDATION (cost minimization)
     The model shows `num_stops` and `distance_km` are large drivers of
     delivery time. We simulate consolidating orders that are geographically
     close (multi-stop batching) vs. dispatching them individually, and
     quantify the time & fuel savings.

Outputs:
  - results/optimization_summary.csv
  - data/optimization_results.csv
  - visualizations/06_optimization_comparison.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

TARGET = "delivery_time_minutes"
SLA_MINUTES = 35

model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

raw_df = pd.read_csv(os.path.join(DATA_DIR, "raw_logistics_data.csv")).drop_duplicates(
    subset=[c for c in ["order_id"] if False] or None
)
# Reload cleaned (already deduplicated / imputed / encoded) frame for consistent features
clean_df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_logistics_data.csv"))

# Recover a "readable" copy (pre-encoding) for grouping by zone / traffic / time
raw_readable = pd.read_csv(os.path.join(DATA_DIR, "raw_logistics_data.csv")).drop_duplicates(
    subset=[c for c in pd.read_csv(os.path.join(DATA_DIR, "raw_logistics_data.csv")).columns if c != "order_id"]
)
raw_readable["weather"] = raw_readable["weather"].fillna(raw_readable["weather"].mode()[0])
raw_readable["rider_rating"] = raw_readable["rider_rating"].fillna(raw_readable["rider_rating"].median())

# Predict delivery time for every historical order using the trained model
X_full = clean_df.drop(columns=[TARGET])[feature_names]
predicted_full = model.predict(X_full)
raw_readable = raw_readable.reset_index(drop=True)
raw_readable["predicted_delivery_time_minutes"] = np.round(predicted_full, 2)

# ---------------------------------------------------------------------------
# STRATEGY 1: Zone-based rider allocation
# ---------------------------------------------------------------------------
# Assume order arrival rate (orders/hour) scales with historical order volume
# share per segment out of an illustrative citywide demand of 480 orders/hour.
TOTAL_ORDERS_PER_HOUR = 480

zone_group = (
    raw_readable.groupby(["area_type", "traffic_level", "time_of_day"])
    .agg(
        avg_predicted_delivery_time=("predicted_delivery_time_minutes", "mean"),
        order_count=("predicted_delivery_time_minutes", "count"),
    )
    .reset_index()
)
zone_group["demand_share"] = zone_group["order_count"] / zone_group["order_count"].sum()
zone_group["orders_per_hour"] = np.round(zone_group["demand_share"] * TOTAL_ORDERS_PER_HOUR, 1)

# Little's Law: riders_needed = arrival_rate (orders/hr) * avg_service_time (hr)
# A small utilization buffer (1.15x) is applied to avoid queue build-up.
zone_group["avg_service_time_hours"] = zone_group["avg_predicted_delivery_time"] / 60
zone_group["riders_needed"] = np.ceil(
    zone_group["orders_per_hour"] * zone_group["avg_service_time_hours"] * 1.15
).astype(int)
zone_group["sla_breach_risk"] = np.where(
    zone_group["avg_predicted_delivery_time"] > SLA_MINUTES, "High", "Low"
)
zone_group = zone_group.sort_values("avg_predicted_delivery_time", ascending=False)

# ---------------------------------------------------------------------------
# STRATEGY 2: Stop-batching / route consolidation simulation
# ---------------------------------------------------------------------------
# Compare: (a) dispatching orders individually (num_stops = 0, one order per trip)
#          (b) batching up to 3 nearby orders per trip (num_stops = 2 extra stops)
# using the trained model to predict time under each scenario, holding all
# other features constant at each order's actual values.

batch_candidates = clean_df.copy()
individual_scenario = batch_candidates.copy()
individual_scenario["num_stops"] = 0

batched_scenario = batch_candidates.copy()
batched_scenario["num_stops"] = np.minimum(batched_scenario["num_stops"] + 2, 4)
# consolidating trims average per-order distance because stops are geographically close
batched_scenario["distance_km"] = batched_scenario["distance_km"] * 0.72

pred_individual = model.predict(individual_scenario.drop(columns=[TARGET])[feature_names])
pred_batched = model.predict(batched_scenario.drop(columns=[TARGET])[feature_names])

# Per-trip time for batched scenario covers up to 3 orders -> normalize to a
# per-order-equivalent time to make it comparable with the individual case.
orders_per_batched_trip = 3
time_per_order_individual = pred_individual
time_per_order_batched = pred_batched / orders_per_batched_trip

avg_time_individual = time_per_order_individual.mean()
avg_time_batched = time_per_order_batched.mean()
time_saving_pct = (avg_time_individual - avg_time_batched) / avg_time_individual * 100

# Approximate fuel usage: ~0.028 L/km for a scooter/motorcycle
FUEL_L_PER_KM = 0.028
FUEL_PRICE_INR = 106  # indicative petrol price per litre (India, illustrative)

total_orders = len(clean_df)
distance_individual_total = clean_df["distance_km"].sum()
distance_batched_total = (clean_df["distance_km"] * 0.72 / orders_per_batched_trip).sum()

fuel_individual_l = distance_individual_total * FUEL_L_PER_KM
fuel_batched_l = distance_batched_total * FUEL_L_PER_KM
fuel_saving_l = fuel_individual_l - fuel_batched_l
cost_saving_inr = fuel_saving_l * FUEL_PRICE_INR

optimization_summary = pd.DataFrame(
    [
        {
            "strategy": "Individual Dispatch (baseline)",
            "avg_delivery_time_per_order_min": round(avg_time_individual, 2),
            "total_fuel_liters": round(fuel_individual_l, 1),
            "estimated_fuel_cost_inr": round(fuel_individual_l * FUEL_PRICE_INR, 0),
        },
        {
            "strategy": "Batched Dispatch (up to 3 orders/trip)",
            "avg_delivery_time_per_order_min": round(avg_time_batched, 2),
            "total_fuel_liters": round(fuel_batched_l, 1),
            "estimated_fuel_cost_inr": round(fuel_batched_l * FUEL_PRICE_INR, 0),
        },
    ]
)
optimization_summary["time_saving_pct_vs_baseline"] = [
    0.0,
    round(time_saving_pct, 2),
]
optimization_summary["fuel_cost_saving_inr_vs_baseline"] = [
    0.0,
    round(cost_saving_inr, 0),
]

opt_summary_path = os.path.join(RESULTS_DIR, "optimization_summary.csv")
optimization_summary.to_csv(opt_summary_path, index=False)
print(f"Optimization summary saved to: {opt_summary_path}")
print(optimization_summary.to_string(index=False))

print(f"\nEstimated per-order delivery time reduction from batching: {time_saving_pct:.1f}%")
print(f"Estimated fuel cost saving across dataset ({total_orders} orders): INR {cost_saving_inr:,.0f}")

# Save the detailed zone allocation table as the "optimization_results.csv" deliverable
zone_out_path = os.path.join(DATA_DIR, "optimization_results.csv")
zone_group.to_csv(zone_out_path, index=False)
print(f"\nZone-based rider allocation table saved to: {zone_out_path}")
print(zone_group.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Visualization: optimization comparison (time & fuel cost)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].bar(
    optimization_summary["strategy"],
    optimization_summary["avg_delivery_time_per_order_min"],
    color=["#A23B72", "#2E86AB"],
)
axes[0].set_ylabel("Avg. Delivery Time per Order (min)")
axes[0].set_title("Delivery Time: Individual vs Batched Dispatch")
axes[0].tick_params(axis="x", rotation=12)

axes[1].bar(
    optimization_summary["strategy"],
    optimization_summary["estimated_fuel_cost_inr"],
    color=["#A23B72", "#2E86AB"],
)
axes[1].set_ylabel("Estimated Fuel Cost (INR)")
axes[1].set_title("Fuel Cost: Individual vs Batched Dispatch")
axes[1].tick_params(axis="x", rotation=12)

fig.suptitle("Route Consolidation / Stop-Batching Optimization Impact")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "06_optimization_comparison.png"), dpi=150)
plt.close()

print("\nOptimization comparison visualization saved to ../visualizations/06_optimization_comparison.png")
