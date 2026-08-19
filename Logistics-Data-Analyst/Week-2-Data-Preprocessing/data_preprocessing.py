"""
data_preprocessing.py
----------------------
Week 2 Task — Data Collection, Cleaning, and Preprocessing for Logistics Analysis
TwoWheel Express last-mile delivery dataset (Thane–Mumbai corridor)

This script implements the full cleaning and preprocessing pipeline described
in Data_Preprocessing_Report.docx:

  1. Handling missing values (drop non-recoverable rows, median/mode impute)
  2. Removing duplicate records and standardizing categorical labels
  3. Outlier detection and treatment (IQR-based capping / removal)
  4. Timestamp and unit validation
  5. Feature scaling / normalization (StandardScaler)

Usage:
    python data_preprocessing.py

Input:
    data/logistics_raw.csv

Output:
    data_clean/logistics_clean.csv   (fully cleaned + scaled, ready for modelling)
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = os.path.join("data", "logistics_raw.csv")
CLEAN_DIR = "data_clean"
CLEAN_PATH = os.path.join(CLEAN_DIR, "logistics_clean.csv")

VEHICLE_MAP = {
    "bike": "Motorcycle",
    "motorbike": "Motorcycle",
    "motorcycle": "Motorcycle",
    "2-wheeler": "Scooter",
    "scooter": "Scooter",
    "e-scooter": "Electric Scooter",
    "electric scooter": "Electric Scooter",
}

NUMERIC_FEATURES = ["distance_km", "order_weight_kg", "rider_rating"]


def iqr_bounds(series, k=1.5):
    """Return (lower, upper) fences using the Interquartile Range method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def handle_missing_values(df):
    """Step 1: drop non-recoverable rows, impute the rest field-by-field."""
    before = len(df)
    df = df.dropna(subset=["order_id", "pickup_time"]).copy()
    dropped_ids = before - len(df)

    num_cols = ["rider_rating", "order_weight_kg"]
    cat_cols = ["weather", "traffic_level"]

    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    print(f"  - Dropped {dropped_ids} rows missing order_id/pickup_time")
    print(f"  - Imputed {num_cols} with median, {cat_cols} with mode")
    return df


def remove_duplicates_and_standardize(df):
    """Step 2: drop duplicate orders, normalize categorical text labels."""
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first").copy()
    print(f"  - Removed {before - len(df)} duplicate order records")

    df["vehicle_type"] = df["vehicle_type"].astype(str).str.strip().str.lower()
    unmapped = set(df["vehicle_type"]) - set(VEHICLE_MAP.keys())
    unmapped = {v for v in unmapped if v not in ("scooter", "motorcycle", "electric scooter")}
    if unmapped:
        print(f"  - Warning: unmapped vehicle_type labels found: {unmapped}")
    df["vehicle_type"] = df["vehicle_type"].map(VEHICLE_MAP).fillna(df["vehicle_type"])
    print(f"  - Standardized vehicle_type to {sorted(df['vehicle_type'].unique())}")

    for col in ["traffic_level", "weather", "zone"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    return df


def treat_outliers(df):
    """Step 3: remove physically impossible values, cap statistical outliers via IQR."""
    before = len(df)
    df = df[(df["delivery_time_min"] > 0) & (df["distance_km"] > 0)].copy()
    print(f"  - Removed {before - len(df)} rows with non-positive delivery_time_min/distance_km")

    for col in ["delivery_time_min", "distance_km"]:
        lower, upper = iqr_bounds(df[col])
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  - Capped {n_capped} outlier values in '{col}' to [{lower:.2f}, {upper:.2f}]")

    return df


def validate_timestamps_and_units(df):
    """Step 4: recompute delivery time from valid timestamps, fix distance units."""
    df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
    df["drop_time"] = pd.to_datetime(df["drop_time"], errors="coerce")

    valid_order = df["drop_time"] > df["pickup_time"]
    n_invalid = (~valid_order).sum()
    df.loc[valid_order, "delivery_time_min"] = (
        (df.loc[valid_order, "drop_time"] - df.loc[valid_order, "pickup_time"]).dt.total_seconds() / 60
    )
    df = df[valid_order].copy()
    print(f"  - Removed {n_invalid} rows with drop_time <= pickup_time")

    suspect_miles = df["distance_km"] > 15
    n_unit_fixed = suspect_miles.sum()
    df.loc[suspect_miles, "distance_km"] = (df.loc[suspect_miles, "distance_km"] * 1.60934).round(2)
    print(f"  - Converted {n_unit_fixed} suspected mile-based distance values to km")

    return df


def scale_features(df):
    """Step 5: standardize numeric features for downstream modelling."""
    preprocess = ColumnTransformer([("scale", StandardScaler(), NUMERIC_FEATURES)])
    pipeline = Pipeline([("preprocess", preprocess)])

    scaled = pipeline.fit_transform(df[NUMERIC_FEATURES])
    df_scaled = df.copy()
    df_scaled[NUMERIC_FEATURES] = scaled
    print(f"  - Standardized {NUMERIC_FEATURES} (mean=0, std=1)")
    return df_scaled


def summarize(before_df, after_df):
    print("\n=== Before vs. After Cleaning ===")
    print(f"{'Metric':35s} {'Before':>15s} {'After':>15s}")
    print(f"{'Total records':35s} {len(before_df):>15,} {len(after_df):>15,}")

    before_missing = before_df.isna().mean().mean() * 100
    after_missing = after_df.isna().mean().mean() * 100
    print(f"{'Missing values (avg % of fields)':35s} {before_missing:>14.1f}% {after_missing:>14.1f}%")

    before_dupes = before_df.duplicated(subset="order_id").sum()
    after_dupes = after_df.duplicated(subset="order_id").sum()
    print(f"{'Duplicate order records':35s} {before_dupes:>15,} {after_dupes:>15,}")

    before_variants = before_df["vehicle_type"].astype(str).str.strip().str.lower().nunique()
    after_variants = after_df["vehicle_type"].nunique()
    print(f"{'vehicle_type label variants':35s} {before_variants:>15,} {after_variants:>15,}")

    print(f"{'Mean delivery_time_min':35s} {before_df['delivery_time_min'].mean():>14.1f} "
          f"{after_df['delivery_time_min'].mean():>14.1f} min")
    print(f"{'Std. dev delivery_time_min':35s} {before_df['delivery_time_min'].std():>14.1f} "
          f"{after_df['delivery_time_min'].std():>14.1f} min")


def main():
    print(f"Loading raw dataset from {RAW_PATH} ...")
    raw_df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(raw_df)} raw records\n")

    df = raw_df.copy()

    print("Step 1: Handling missing values")
    df = handle_missing_values(df)

    print("\nStep 2: Removing duplicates & standardizing categories")
    df = remove_duplicates_and_standardize(df)

    print("\nStep 3: Outlier detection and treatment (IQR)")
    df = treat_outliers(df)

    print("\nStep 4: Timestamp and unit validation")
    df = validate_timestamps_and_units(df)

    print("\nStep 5: Feature scaling / normalization")
    df_scaled = scale_features(df)

    os.makedirs(CLEAN_DIR, exist_ok=True)
    df_scaled.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved final cleaned + scaled dataset -> {CLEAN_PATH}")

    summarize(raw_df, df)


if __name__ == "__main__":
    main()
