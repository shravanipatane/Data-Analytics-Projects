"""
01_problem_definition_data_simulation.py

PROBLEM DEFINITION
-------------------
Business context : A two-wheeler (motorbike/scooter) based last-mile logistics
                    company (food, grocery, parcel and e-commerce deliveries)
                    wants to forecast DELIVERY TIME (in minutes) for every new
                    order at the moment it is assigned to a rider, so that:
                      1. Customers receive an accurate ETA.
                      2. Dispatch can decide which rider/zone to assign the
                         order to.
                      3. Operations can identify bottlenecks (traffic, weather,
                         rider load, distance) and optimize fleet allocation.

Target variable  : delivery_time_minutes  (continuous -> regression problem)

Features (raw)   :
  order_id                  - unique id
  distance_km                - pickup-to-drop distance (km)
  traffic_level               - Low / Medium / High / Severe (categorical)
  weather                     - Clear / Rain / Fog / Extreme_Heat (categorical)
  time_of_day                 - Morning / Afternoon / Evening / Night (categorical)
  day_of_week                 - Mon..Sun (categorical)
  area_type                   - Urban / Suburban / Rural (categorical)
  package_weight_kg            - weight of parcel/food bag (kg)
  num_stops                   - number of stops rider has to make before drop
  rider_experience_years        - rider's experience (years)
  rider_rating                 - rider's average customer rating (1-5)
  vehicle_type                 - Scooter / Motorcycle / Electric_Two_Wheeler
  is_peak_hour                 - 1/0 flag for lunch/dinner or office peak hours
  fuel_level_pct                - fuel/battery level at pickup (%)
  delivery_time_minutes          - TARGET (minutes taken pickup -> drop)

This script SIMULATES a realistic dataset (n = 5,000 orders) with plausible
statistical relationships (e.g. delivery time increases with distance,
traffic, number of stops and bad weather; decreases slightly with rider
experience). Random noise is injected so the problem is non-trivial - this
mirrors a real-world operations dataset that would normally be pulled from a
company's order-management / GPS-tracking system.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 5000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Simulate raw features
# ---------------------------------------------------------------------------
traffic_levels = np.random.choice(
    ["Low", "Medium", "High", "Severe"], size=N, p=[0.30, 0.35, 0.25, 0.10]
)
weather_options = np.random.choice(
    ["Clear", "Rain", "Fog", "Extreme_Heat"], size=N, p=[0.60, 0.20, 0.10, 0.10]
)
time_of_day = np.random.choice(
    ["Morning", "Afternoon", "Evening", "Night"], size=N, p=[0.25, 0.30, 0.30, 0.15]
)
day_of_week = np.random.choice(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], size=N
)
area_type = np.random.choice(
    ["Urban", "Suburban", "Rural"], size=N, p=[0.55, 0.35, 0.10]
)
vehicle_type = np.random.choice(
    ["Scooter", "Motorcycle", "Electric_Two_Wheeler"], size=N, p=[0.45, 0.40, 0.15]
)

distance_km = np.round(np.random.gamma(shape=2.2, scale=1.8, size=N) + 0.5, 2)
distance_km = np.clip(distance_km, 0.5, 25)

package_weight_kg = np.round(np.random.exponential(scale=2.0, size=N) + 0.2, 2)
package_weight_kg = np.clip(package_weight_kg, 0.2, 20)

num_stops = np.random.poisson(lam=0.6, size=N)
num_stops = np.clip(num_stops, 0, 4)

rider_experience_years = np.round(np.random.exponential(scale=2.5, size=N), 1)
rider_experience_years = np.clip(rider_experience_years, 0, 15)

rider_rating = np.round(np.clip(np.random.normal(4.3, 0.4, size=N), 2.5, 5.0), 2)

is_peak_hour = np.where(
    np.isin(time_of_day, ["Afternoon", "Evening"]) & (np.random.rand(N) < 0.55), 1, 0
)

fuel_level_pct = np.round(np.random.uniform(10, 100, size=N), 1)

order_id = [f"ORD{100000+i}" for i in range(N)]

# ---------------------------------------------------------------------------
# 2. Simulate target: delivery_time_minutes using a plausible generative model
# ---------------------------------------------------------------------------
traffic_penalty = {"Low": 0, "Medium": 4, "High": 9, "Severe": 16}
weather_penalty = {"Clear": 0, "Rain": 6, "Fog": 8, "Extreme_Heat": 3}
area_speed_factor = {"Urban": 1.15, "Suburban": 1.0, "Rural": 0.85}
peak_penalty = np.where(is_peak_hour == 1, 5, 0)

base_time = 6.0  # fixed pickup/handover overhead (minutes)
speed_component = distance_km * 3.1  # base ~ 3.1 min/km on a two-wheeler in mixed traffic

traffic_component = np.array([traffic_penalty[t] for t in traffic_levels])
weather_component = np.array([weather_penalty[w] for w in weather_options])
area_component = np.array([area_speed_factor[a] for a in area_type])

stop_penalty = num_stops * 3.5
weight_penalty = np.where(package_weight_kg > 5, (package_weight_kg - 5) * 0.6, 0)
experience_bonus = -np.minimum(rider_experience_years * 0.35, 4.5)  # experienced riders are faster
fuel_penalty = np.where(fuel_level_pct < 20, 2.0, 0)

noise = np.random.normal(0, 3.2, size=N)

delivery_time_minutes = (
    base_time
    + speed_component * area_component
    + traffic_component
    + weather_component
    + stop_penalty
    + weight_penalty
    + experience_bonus
    + peak_penalty
    + fuel_penalty
    + noise
)
delivery_time_minutes = np.round(np.clip(delivery_time_minutes, 5, 120), 2)

# ---------------------------------------------------------------------------
# 3. Assemble DataFrame and inject light missingness / duplicate noise
#    (to emulate a realistic "raw" operational export that needs cleaning)
# ---------------------------------------------------------------------------
df = pd.DataFrame(
    {
        "order_id": order_id,
        "distance_km": distance_km,
        "traffic_level": traffic_levels,
        "weather": weather_options,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "area_type": area_type,
        "package_weight_kg": package_weight_kg,
        "num_stops": num_stops,
        "rider_experience_years": rider_experience_years,
        "rider_rating": rider_rating,
        "vehicle_type": vehicle_type,
        "is_peak_hour": is_peak_hour,
        "fuel_level_pct": fuel_level_pct,
        "delivery_time_minutes": delivery_time_minutes,
    }
)

# Inject ~1.5% missing values into a couple of columns to simulate real data
missing_idx_weather = np.random.choice(df.index, size=int(0.015 * N), replace=False)
df.loc[missing_idx_weather, "weather"] = np.nan
missing_idx_rating = np.random.choice(df.index, size=int(0.01 * N), replace=False)
df.loc[missing_idx_rating, "rider_rating"] = np.nan

# Duplicate a handful of rows (data-entry duplication) to be cleaned in step 02
dup_rows = df.sample(10, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

raw_path = os.path.join(OUT_DIR, "raw_logistics_data.csv")
df.to_csv(raw_path, index=False)

print(f"Simulated raw dataset saved to: {raw_path}")
print(f"Shape: {df.shape}")
print(df.head())
