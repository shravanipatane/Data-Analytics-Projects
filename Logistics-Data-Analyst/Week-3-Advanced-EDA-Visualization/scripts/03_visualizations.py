"""
03_visualizations.py
---------------------
Generates all visualizations for the Two-Wheeler Logistics EDA report.

Each chart is chosen deliberately for the question it answers:
    01_delivery_time_dist.png   -> Histogram: shape/spread of delivery times
    02_delaybox_traffic.png     -> Boxplot: delivery time spread by traffic
    03_distance_vs_time.png     -> Scatter: distance-time relationship by weather
    04_avg_cost_vehicle.png     -> Bar: cost efficiency across vehicle types
    05_correlation_heatmap.png  -> Heatmap: relationships among numeric metrics
    06_delay_rate_zone.png      -> Bar: bottleneck zones by delay rate
    07_daily_trend.png          -> Line: operational trend over time
    08_vehicle_share_pie.png    -> Pie: fleet composition
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

OUT = "../visualizations/"
df = pd.read_csv("../data/two_wheeler_logistics.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------------
# 1. Histogram - Delivery Time Distribution
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["delivery_time_min"], bins=35, kde=True, color="#2E86AB", ax=ax)
ax.axvline(df["delivery_time_min"].mean(), color="#E63946", linestyle="--",
           label=f"Mean = {df['delivery_time_min'].mean():.1f} min")
ax.axvline(df["delivery_time_min"].median(), color="#F4A261", linestyle="--",
           label=f"Median = {df['delivery_time_min'].median():.1f} min")
ax.set_title("Distribution of Delivery Times")
ax.set_xlabel("Delivery Time (minutes)")
ax.set_ylabel("Number of Deliveries")
ax.legend()
plt.tight_layout()
plt.savefig(OUT + "01_delivery_time_dist.png")
plt.close()

# ---------------------------------------------------------------
# 2. Boxplot - Delivery Time by Traffic Condition
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
order = ["Low", "Medium", "High"]
sns.boxplot(data=df, x="traffic_condition", y="delivery_time_min",
            order=order, hue="traffic_condition", palette="YlOrRd", legend=False, ax=ax)
ax.set_title("Delivery Time Spread by Traffic Condition")
ax.set_xlabel("Traffic Condition")
ax.set_ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig(OUT + "02_delaybox_traffic.png")
plt.close()

# ---------------------------------------------------------------
# 3. Scatter - Distance vs Delivery Time, colored by Weather
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(data=df, x="distance_km", y="delivery_time_min",
                 hue="weather", palette={"Clear": "#2E86AB", "Cloudy": "#8D99AE", "Rain": "#E63946"},
                 alpha=0.6, s=35, ax=ax)
ax.set_title("Distance vs. Delivery Time by Weather Condition")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Delivery Time (minutes)")
ax.legend(title="Weather")
plt.tight_layout()
plt.savefig(OUT + "03_distance_vs_time.png")
plt.close()

# ---------------------------------------------------------------
# 4. Bar - Average Delivery Cost & Fuel Cost by Vehicle Type
# ---------------------------------------------------------------
grp = df.groupby("vehicle_type")[["fuel_cost_inr", "delivery_cost_inr"]].mean().reindex(
    ["Scooter", "Motorbike", "E-Bike"])
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(grp))
width = 0.35
ax.bar(x - width/2, grp["fuel_cost_inr"], width, label="Avg Fuel Cost (₹)", color="#F4A261")
ax.bar(x + width/2, grp["delivery_cost_inr"], width, label="Avg Total Delivery Cost (₹)", color="#2E86AB")
ax.set_xticks(x)
ax.set_xticklabels(grp.index)
ax.set_title("Average Cost by Vehicle Type")
ax.set_ylabel("Cost (INR)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT + "04_avg_cost_vehicle.png")
plt.close()

# ---------------------------------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------------------------------
numeric_cols = ["distance_km", "shipment_weight_kg", "delivery_time_min",
                 "fuel_cost_inr", "delivery_cost_inr", "customer_rating"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix - Logistics Metrics")
plt.tight_layout()
plt.savefig(OUT + "05_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 6. Bar - Delay Rate by Zone (bottleneck identification)
# ---------------------------------------------------------------
delay_zone = (df.groupby("zone")["delayed"].mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#E63946" if v == delay_zone.max() else "#2E86AB" for v in delay_zone]
ax.barh(delay_zone.index, delay_zone.values, color=colors)
ax.set_title("Delay Rate (%) by Delivery Zone")
ax.set_xlabel("Delay Rate (%)")
ax.invert_yaxis()
for i, v in enumerate(delay_zone.values):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center")
plt.tight_layout()
plt.savefig(OUT + "06_delay_rate_zone.png")
plt.close()

# ---------------------------------------------------------------
# 7. Line - Daily Trend: Avg Delivery Time & Volume over Time
# ---------------------------------------------------------------
daily = df.groupby("date").agg(avg_time=("delivery_time_min", "mean"),
                                 volume=("delivery_id", "count")).reset_index()
daily["avg_time_7d"] = daily["avg_time"].rolling(7, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(daily["date"], daily["avg_time"], color="#8D99AE", alpha=0.4, label="Daily Avg (raw)")
ax1.plot(daily["date"], daily["avg_time_7d"], color="#E63946", linewidth=2.2, label="7-Day Rolling Avg")
ax1.set_ylabel("Avg Delivery Time (min)")
ax1.set_xlabel("Date")
ax1.set_title("Daily Average Delivery Time Trend (May-Jul 2026)")
ax1.legend(loc="upper left")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT + "07_daily_trend.png")
plt.close()

# ---------------------------------------------------------------
# 8. Pie - Fleet Composition (Vehicle Type Share)
# ---------------------------------------------------------------
share = df["vehicle_type"].value_counts()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
colors_pie = ["#2E86AB", "#F4A261", "#2A9D8F"]
ax.pie(share.values, labels=share.index, autopct="%1.1f%%", startangle=90,
       colors=colors_pie, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Fleet Composition by Vehicle Type")
plt.tight_layout()
plt.savefig(OUT + "08_vehicle_share_pie.png")
plt.close()

print("All 8 visualizations saved to", OUT)
import os
for f in sorted(os.listdir(OUT)):
    print(" -", f)
