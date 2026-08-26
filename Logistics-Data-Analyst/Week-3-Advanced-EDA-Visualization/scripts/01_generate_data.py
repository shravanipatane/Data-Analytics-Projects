"""
01_generate_data.py
--------------------
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Domain    : Two-Wheeler (Bike/Scooter) Last-Mile Delivery Logistics

Purpose:
    Simulate a realistic two-wheeler logistics dataset for a last-mile
    delivery operation (e.g. food / grocery / parcel delivery via
    bikes, scooters and e-bikes). The simulated relationships mimic
    real-world logistics behaviour so that the downstream EDA and
    visualizations produce meaningful, defensible insights.

Output:
    ../data/two_wheeler_logistics.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 2000  # number of delivery records

# ---------------------------------------------------------------
# 1. Base categorical fields
# ---------------------------------------------------------------
vehicle_types = np.random.choice(
    ["Scooter", "Motorbike", "E-Bike"], size=N, p=[0.45, 0.35, 0.20]
)

zones = np.random.choice(
    ["Zone A - Central", "Zone B - North", "Zone C - South",
     "Zone D - East", "Zone E - West"],
    size=N, p=[0.28, 0.20, 0.18, 0.17, 0.17]
)

weather = np.random.choice(
    ["Clear", "Cloudy", "Rain"], size=N, p=[0.62, 0.23, 0.15]
)

traffic = np.random.choice(
    ["Low", "Medium", "High"], size=N, p=[0.30, 0.45, 0.25]
)

time_of_day = np.random.choice(
    ["Morning (6-11)", "Afternoon (11-16)", "Evening (16-21)", "Night (21-2)"],
    size=N, p=[0.22, 0.28, 0.35, 0.15]
)

riders = [f"RDR-{i:03d}" for i in range(1, 61)]
rider_id = np.random.choice(riders, size=N)

# random dates across a 90-day operating window
start_date = datetime(2026, 5, 1)
dates = [start_date + timedelta(days=int(d)) for d in np.random.randint(0, 90, size=N)]

# ---------------------------------------------------------------
# 2. Continuous / derived fields with realistic dependencies
# ---------------------------------------------------------------
distance_km = np.round(np.random.gamma(shape=2.2, scale=1.7, size=N) + 0.5, 2)
distance_km = np.clip(distance_km, 0.5, 18)

shipment_weight_kg = np.round(np.random.gamma(shape=2.0, scale=1.1, size=N) + 0.2, 2)
shipment_weight_kg = np.clip(shipment_weight_kg, 0.2, 12)

# base speed (km/h) depends on traffic & weather
traffic_penalty = {"Low": 1.0, "Medium": 1.35, "High": 1.85}
weather_penalty = {"Clear": 1.0, "Cloudy": 1.08, "Rain": 1.45}
vehicle_speed_factor = {"Scooter": 1.0, "Motorbike": 0.88, "E-Bike": 1.12}

base_speed_kmph = 26
delivery_time_min = []
for i in range(N):
    speed = base_speed_kmph / (traffic_penalty[traffic[i]] * weather_penalty[weather[i]]
                                * vehicle_speed_factor[vehicle_types[i]])
    travel_time = (distance_km[i] / speed) * 60
    handling_time = np.random.normal(6, 2)          # pickup/drop-off handling
    noise = np.random.normal(0, 3)
    t = travel_time + max(handling_time, 1) + noise
    delivery_time_min.append(max(round(t, 1), 4))
delivery_time_min = np.array(delivery_time_min)

# cost model: fuel + distance + weight surcharge
fuel_rate_per_km = {"Scooter": 1.8, "Motorbike": 2.3, "E-Bike": 0.6}
fuel_cost = np.array([
    round(distance_km[i] * fuel_rate_per_km[vehicle_types[i]] * np.random.uniform(0.9, 1.1), 2)
    for i in range(N)
])
weight_surcharge = np.round(np.where(shipment_weight_kg > 5, (shipment_weight_kg - 5) * 3, 0), 2)
delivery_cost_inr = np.round(25 + fuel_cost + weight_surcharge
                              + np.random.normal(0, 3, N), 2)
delivery_cost_inr = np.clip(delivery_cost_inr, 20, None)

# promised SLA (minutes) varies slightly by zone type (central = tighter SLA)
sla_map = {"Zone A - Central": 30, "Zone B - North": 35, "Zone C - South": 35,
           "Zone D - East": 40, "Zone E - West": 40}
sla_minutes = np.array([sla_map[z] for z in zones])
delayed = (delivery_time_min > sla_minutes).astype(int)

# customer rating (1-5) penalized by delay & rain, small noise
rating = (5
          - delayed * np.random.uniform(0.8, 1.6, N)
          - (weather == "Rain") * np.random.uniform(0.1, 0.4, N)
          + np.random.normal(0, 0.25, N))
customer_rating = np.clip(np.round(rating, 1), 1.0, 5.0)

order_volume = np.random.choice([1, 2, 3, 4], size=N, p=[0.55, 0.28, 0.12, 0.05])

# ---------------------------------------------------------------
# 3. Assemble DataFrame
# ---------------------------------------------------------------
df = pd.DataFrame({
    "delivery_id": [f"DEL-{10000+i}" for i in range(N)],
    "date": [d.strftime("%Y-%m-%d") for d in dates],
    "rider_id": rider_id,
    "vehicle_type": vehicle_types,
    "zone": zones,
    "time_of_day": time_of_day,
    "weather": weather,
    "traffic_condition": traffic,
    "distance_km": distance_km,
    "shipment_weight_kg": shipment_weight_kg,
    "order_volume_parcels": order_volume,
    "sla_minutes": sla_minutes,
    "delivery_time_min": delivery_time_min,
    "delayed": delayed,
    "fuel_cost_inr": fuel_cost,
    "delivery_cost_inr": delivery_cost_inr,
    "customer_rating": customer_rating,
})

df = df.sort_values("date").reset_index(drop=True)
df.to_csv("../data/two_wheeler_logistics.csv", index=False)

print(f"Generated {len(df)} records -> ../data/two_wheeler_logistics.csv")
print(df.head())
print("\nOverall on-time rate: {:.1f}%".format(100 * (1 - df['delayed'].mean())))
