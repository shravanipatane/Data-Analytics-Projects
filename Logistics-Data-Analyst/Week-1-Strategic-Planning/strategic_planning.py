"""
Two-Wheeler Logistics — Week 1
Strategic Planning & Data Exploration

Project:
Optimizing Two-Wheeler Last-Mile Delivery Operations
in the Thane–Mumbai Corridor

This script provides the core Python implementation described in
the Week 1 Strategic Planning Report:
- synthetic logistics data generation
- data cleaning
- KPI calculation
- exploratory analysis
- delivery-time prediction
- zone clustering
- nearest-neighbour route optimization

Dataset: synthetic but realistic 1,200-order logistics dataset.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ORDERS = 1200

rng = np.random.default_rng(RANDOM_STATE)

zones = [
    "Thane West",
    "Thane East",
    "Ghodbunder Road",
    "Mulund",
    "Bhandup",
    "Vikhroli",
    "Kalyan",
    "Dombivli",
    "Airoli",
    "Vashi",
]

vehicle_types = ["Petrol Scooter", "Motorcycle", "Electric Scooter"]
traffic_levels = ["Low", "Medium", "High"]
weather_types = ["Clear", "Cloudy", "Rain"]
time_slots = ["Morning", "Afternoon", "Evening", "Night"]


# ============================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================

def generate_logistics_data(n_orders=1200):
    """Generate a synthetic but realistic two-wheeler logistics dataset."""

    zone_weights = np.array(
        [0.19, 0.11, 0.16, 0.10, 0.07,
         0.08, 0.11, 0.08, 0.05, 0.05]
    )

    df = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "zone": rng.choice(zones, size=n_orders, p=zone_weights),
        "vehicle_type": rng.choice(
            vehicle_types,
            size=n_orders,
            p=[0.45, 0.35, 0.20]
        ),
        "traffic_level": rng.choice(
            traffic_levels,
            size=n_orders,
            p=[0.25, 0.45, 0.30]
        ),
        "weather": rng.choice(
            weather_types,
            size=n_orders,
            p=[0.65, 0.25, 0.10]
        ),
        "time_slot": rng.choice(
            time_slots,
            size=n_orders,
            p=[0.25, 0.25, 0.35, 0.15]
        ),
        "distance_km": rng.gamma(shape=2.4, scale=2.2, size=n_orders),
        "order_weight_kg": rng.gamma(shape=2.0, scale=1.2, size=n_orders),
        "rider_id": rng.integers(1, 101, size=n_orders),
    })

    # Keep distances realistic for last-mile delivery.
    df["distance_km"] = df["distance_km"].clip(0.5, 18)

    # Base delivery time.
    traffic_effect = {
        "Low": 0,
        "Medium": 4,
        "High": 10,
    }

    weather_effect = {
        "Clear": 0,
        "Cloudy": 1.5,
        "Rain": 5,
    }

    vehicle_effect = {
        "Motorcycle": -1.0,
        "Petrol Scooter": 0,
        "Electric Scooter": 0.8,
    }

    time_effect = {
        "Morning": 1,
        "Afternoon": 0,
        "Evening": 5,
        "Night": -1,
    }

    df["delivery_time_min"] = (
        8
        + df["distance_km"] * 2.4
        + df["order_weight_kg"] * 0.45
        + df["traffic_level"].map(traffic_effect)
        + df["weather"].map(weather_effect)
        + df["vehicle_type"].map(vehicle_effect)
        + df["time_slot"].map(time_effect)
        + rng.normal(0, 2.5, n_orders)
    ).clip(5, None)

    # Promised delivery time.
    df["promised_time_min"] = (
        df["delivery_time_min"]
        + rng.normal(4, 2, n_orders)
    ).clip(8, None)

    df["on_time"] = (
        df["delivery_time_min"] <= df["promised_time_min"]
    ).astype(int)

    # Synthetic operating cost.
    cost_rate = {
        "Petrol Scooter": 1.95,
        "Motorcycle": 1.85,
        "Electric Scooter": 1.30,
    }

    df["cost_per_km"] = (
        df["vehicle_type"].map(cost_rate)
        + rng.normal(0, 0.12, n_orders)
    ).clip(0.8, None)

    df["total_cost"] = df["distance_km"] * df["cost_per_km"]

    # Introduce realistic data-quality issues.
    missing_indices = rng.choice(df.index, size=35, replace=False)
    df.loc[missing_indices[:12], "weather"] = np.nan
    df.loc[missing_indices[12:24], "order_weight_kg"] = np.nan
    df.loc[missing_indices[24:], "traffic_level"] = np.nan

    # Add a few duplicated rows.
    duplicates = df.sample(8, random_state=RANDOM_STATE)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Add a few distance outliers.
    outlier_indices = rng.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices, "distance_km"] = 80

    return df


# ============================================================
# 3. DATA CLEANING
# ============================================================

def clean_data(df):
    """Clean missing values, duplicates, text fields, and outliers."""

    cleaned = df.copy()

    # Standardize categorical text.
    categorical_cols = [
        "zone",
        "vehicle_type",
        "traffic_level",
        "weather",
        "time_slot",
    ]

    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    # Remove duplicate order records.
    cleaned = cleaned.drop_duplicates(subset="order_id")

    # Fill missing categorical values with the mode.
    for col in ["traffic_level", "weather"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    # Fill missing numerical values using the median.
    cleaned["order_weight_kg"] = cleaned["order_weight_kg"].fillna(
        cleaned["order_weight_kg"].median()
    )

    # Remove extreme distance outliers using IQR.
    q1 = cleaned["distance_km"].quantile(0.25)
    q3 = cleaned["distance_km"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    cleaned = cleaned[
        cleaned["distance_km"].between(lower, upper)
    ].copy()

    # Recalculate derived fields.
    cleaned["speed_kmh"] = (
        cleaned["distance_km"] / cleaned["delivery_time_min"] * 60
    )

    cleaned["delay_min"] = (
        cleaned["delivery_time_min"]
        - cleaned["promised_time_min"]
    ).clip(lower=0)

    return cleaned.reset_index(drop=True)


# ============================================================
# 4. KPI ANALYSIS
# ============================================================

def calculate_kpis(df):
    """Calculate fleet-wide logistics KPIs."""

    kpis = {
        "On-Time Delivery Rate (%)":
            df["on_time"].mean() * 100,

        "Average Delivery Time (min)":
            df["delivery_time_min"].mean(),

        "Average Cost per Km":
            df["cost_per_km"].mean(),

        "Average Delivery Speed (km/h)":
            df["speed_kmh"].mean(),

        "Average Orders per Rider":
            df.groupby("rider_id")["order_id"].count().mean(),
    }

    return pd.Series(kpis)


def calculate_zone_kpis(df):
    """Calculate operational KPIs by logistics zone."""

    zone_kpis = (
        df.groupby("zone")
        .agg(
            orders=("order_id", "count"),
            avg_delivery_time=("delivery_time_min", "mean"),
            on_time_rate=("on_time", "mean"),
            avg_distance=("distance_km", "mean"),
            avg_cost_per_km=("cost_per_km", "mean"),
        )
        .reset_index()
    )

    zone_kpis["on_time_rate"] *= 100

    return zone_kpis.sort_values(
        "orders", ascending=False
    )


# ============================================================
# 5. PREDICTIVE MODELING
# ============================================================

def build_predictive_models(df):
    """Train Linear Regression and Random Forest models."""

    features_num = [
        "distance_km",
        "order_weight_kg",
    ]

    features_cat = [
        "vehicle_type",
        "traffic_level",
        "weather",
        "time_slot",
    ]

    X = df[features_num + features_cat]
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    preprocess = ColumnTransformer([
        (
            "num",
            "passthrough",
            features_num,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            features_cat,
        ),
    ])

    # Linear Regression.
    linear_model = Pipeline([
        ("prep", preprocess),
        ("reg", LinearRegression()),
    ])

    linear_model.fit(X_train, y_train)

    linear_pred = linear_model.predict(X_test)

    linear_mae = mean_absolute_error(
        y_test,
        linear_pred
    )

    linear_r2 = r2_score(
        y_test,
        linear_pred
    )

    # Random Forest.
    rf_model = Pipeline([
        ("prep", preprocess),
        (
            "reg",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    rf_r2 = r2_score(
        y_test,
        rf_pred
    )

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest Regressor",
        ],
        "MAE (min)": [
            linear_mae,
            rf_mae,
        ],
        "R2 Score": [
            linear_r2,
            rf_r2,
        ],
    })

    return results, rf_model


# ============================================================
# 6. ZONE CLUSTERING
# ============================================================

def cluster_zones(df):
    """Segment zones based on demand and service characteristics."""

    zone_features = (
        df.groupby("zone")
        .agg(
            orders=("order_id", "count"),
            avg_delivery_time=("delivery_time_min", "mean"),
            avg_distance=("distance_km", "mean"),
            on_time_rate=("on_time", "mean"),
        )
        .reset_index()
    )

    feature_cols = [
        "orders",
        "avg_delivery_time",
        "avg_distance",
        "on_time_rate",
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        zone_features[feature_cols]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    zone_features["cluster"] = kmeans.fit_predict(
        X_scaled
    )

    # Label clusters based on order volume.
    cluster_order_volume = (
        zone_features.groupby("cluster")["orders"]
        .mean()
        .sort_values(ascending=False)
    )

    tier_names = ["High-Demand Hub Priority",
                  "Medium-Demand Zone",
                  "Low-Demand / Consolidate"]

    mapping = {
        cluster: tier_names[i]
        for i, cluster in enumerate(
            cluster_order_volume.index
        )
    }

    zone_features["priority_tier"] = (
        zone_features["cluster"].map(mapping)
    )

    return zone_features.sort_values(
        "orders",
        ascending=False
    )


# ============================================================
# 7. ROUTE OPTIMIZATION
# ============================================================

# Approximate coordinates for the Thane–Mumbai corridor.
ZONE_COORDINATES = {
    "Thane West": (19.2183, 72.9781),
    "Thane East": (19.1970, 73.0010),
    "Ghodbunder Road": (19.2600, 72.9700),
    "Mulund": (19.1726, 72.9567),
    "Bhandup": (19.1451, 72.9372),
    "Vikhroli": (19.1111, 72.9278),
    "Kalyan": (19.2437, 73.1355),
    "Dombivli": (19.2183, 73.0867),
    "Airoli": (19.1590, 72.9986),
    "Vashi": (19.0771, 72.9987),
}


def haversine_distance(coord1, coord2):
    """Return approximate distance between two lat/lon points in km."""

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return 6371 * 2 * asin(sqrt(a))


def route_distance(route):
    """Calculate total distance for an ordered route."""

    total = 0

    for i in range(len(route) - 1):
        total += haversine_distance(
            ZONE_COORDINATES[route[i]],
            ZONE_COORDINATES[route[i + 1]],
        )

    return total


def nearest_neighbor_route(start_zone, stop_list):
    """
    Create a route using the nearest-neighbour heuristic.
    """

    route = [start_zone]
    remaining = list(stop_list)

    if start_zone in remaining:
        remaining.remove(start_zone)

    current = start_zone

    while remaining:
        next_stop = min(
            remaining,
            key=lambda zone: haversine_distance(
                ZONE_COORDINATES[current],
                ZONE_COORDINATES[zone],
            ),
        )

        route.append(next_stop)
        current = next_stop
        remaining.remove(next_stop)

    return route


# ============================================================
# 8. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 70)
    print("TWO-WHEELER LOGISTICS — WEEK 1")
    print("Strategic Planning & Data Exploration")
    print("=" * 70)

    # Generate data.
    raw_df = generate_logistics_data(N_ORDERS)

    print("\nRaw dataset shape:", raw_df.shape)
    print("\nRaw dataset preview:")
    print(raw_df.head())

    # Clean data.
    clean_df = clean_data(raw_df)

    print("\nClean dataset shape:", clean_df.shape)

    # KPI analysis.
    print("\n" + "=" * 70)
    print("FLEET-WIDE KPI SUMMARY")
    print("=" * 70)

    kpis = calculate_kpis(clean_df)

    for name, value in kpis.items():
        print(f"{name}: {value:.2f}")

    # Zone analysis.
    print("\n" + "=" * 70)
    print("ZONE KPI ANALYSIS")
    print("=" * 70)

    zone_kpis = calculate_zone_kpis(clean_df)
    print(zone_kpis.round(2).to_string(index=False))

    # Predictive modeling.
    print("\n" + "=" * 70)
    print("PREDICTIVE MODEL PERFORMANCE")
    print("=" * 70)

    model_results, rf_model = build_predictive_models(
        clean_df
    )

    print(model_results.round(3).to_string(index=False))

    # Zone clustering.
    print("\n" + "=" * 70)
    print("ZONE SEGMENTATION")
    print("=" * 70)

    clustered_zones = cluster_zones(clean_df)

    print(
        clustered_zones[
            [
                "zone",
                "orders",
                "avg_delivery_time",
                "priority_tier",
            ]
        ].round(2).to_string(index=False)
    )

    # Route optimization.
    print("\n" + "=" * 70)
    print("ROUTE OPTIMIZATION")
    print("=" * 70)

    sample_route = [
        "Thane West",
        "Vashi",
        "Mulund",
        "Kalyan",
        "Airoli",
    ]

    optimized_route = nearest_neighbor_route(
        start_zone="Thane West",
        stop_list=sample_route,
    )

    original_distance = route_distance(sample_route)
    optimized_distance = route_distance(optimized_route)

    saving_pct = (
        (original_distance - optimized_distance)
        / original_distance
        * 100
    )

    print("Original route:")
    print(" -> ".join(sample_route))

    print("\nOptimized route:")
    print(" -> ".join(optimized_route))

    print(f"\nOriginal distance: {original_distance:.2f} km")
    print(f"Optimized distance: {optimized_distance:.2f} km")
    print(f"Distance reduction: {saving_pct:.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
