"""
02_eda_analysis.py
-------------------
Exploratory Data Analysis (EDA) for the two-wheeler logistics dataset.

Covers:
    - Central tendency (mean, median, mode) for key numeric metrics
    - Spread / dispersion (std dev, IQR)
    - Distribution shape (skewness)
    - Correlation matrix between numeric variables
    - Group-wise aggregates (by vehicle type, zone, traffic, weather)

Output:
    Prints a full report to stdout AND saves it to
    ../report/eda_summary.txt for inclusion in the Word report.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("../data/two_wheeler_logistics.csv")

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

log("=" * 70)
log("TWO-WHEELER LOGISTICS DATASET - EXPLORATORY DATA ANALYSIS")
log("=" * 70)
log(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Date range: {df['date'].min()} to {df['date'].max()}")

numeric_cols = ["distance_km", "shipment_weight_kg", "delivery_time_min",
                 "fuel_cost_inr", "delivery_cost_inr", "customer_rating"]

log("\n--- 1. CENTRAL TENDENCY & DISPERSION ---")
desc = df[numeric_cols].describe().T
desc["median"] = df[numeric_cols].median()
desc["skew"] = df[numeric_cols].skew()
desc["mode"] = df[numeric_cols].mode().iloc[0]
log(desc.round(2).to_string())

log("\n--- 2. MISSING VALUES CHECK ---")
log(df.isnull().sum().to_string())

log("\n--- 3. ON-TIME PERFORMANCE ---")
delay_rate = df["delayed"].mean() * 100
log(f"Overall delay rate: {delay_rate:.2f}%")
log(f"Overall on-time rate: {100 - delay_rate:.2f}%")

log("\n--- 4. GROUP-WISE AVERAGES: VEHICLE TYPE ---")
log(df.groupby("vehicle_type")[["delivery_time_min", "fuel_cost_inr",
                                  "delivery_cost_inr", "delayed"]].mean().round(2).to_string())

log("\n--- 5. GROUP-WISE AVERAGES: TRAFFIC CONDITION ---")
log(df.groupby("traffic_condition")[["delivery_time_min", "delayed",
                                       "customer_rating"]].mean().round(2)
    .reindex(["Low", "Medium", "High"]).to_string())

log("\n--- 6. GROUP-WISE AVERAGES: WEATHER ---")
log(df.groupby("weather")[["delivery_time_min", "delayed",
                             "customer_rating"]].mean().round(2).to_string())

log("\n--- 7. GROUP-WISE AVERAGES: ZONE ---")
log(df.groupby("zone")[["delivery_time_min", "delayed", "delivery_cost_inr"]]
    .mean().round(2).to_string())

log("\n--- 8. CORRELATION MATRIX (numeric variables) ---")
corr = df[numeric_cols].corr()
log(corr.round(2).to_string())

log("\n--- 9. KEY CORRELATION HIGHLIGHTS ---")
log(f"distance_km vs delivery_time_min : r = {corr.loc['distance_km','delivery_time_min']:.2f}")
log(f"distance_km vs fuel_cost_inr     : r = {corr.loc['distance_km','fuel_cost_inr']:.2f}")
log(f"delivery_time_min vs rating      : r = {corr.loc['delivery_time_min','customer_rating']:.2f}")
log(f"shipment_weight vs delivery_cost : r = {corr.loc['shipment_weight_kg','delivery_cost_inr']:.2f}")

log("\n--- 10. TOP 5 RIDERS BY DELIVERY VOLUME ---")
log(df["rider_id"].value_counts().head(5).to_string())

log("\n--- 11. VEHICLE TYPE SHARE ---")
log((df["vehicle_type"].value_counts(normalize=True) * 100).round(1).to_string())

with open("../report/eda_summary.txt", "w") as f:
    f.write("\n".join(lines))

print("\nSaved full EDA text report -> ../report/eda_summary.txt")
