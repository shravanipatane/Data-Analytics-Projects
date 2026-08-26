"""
02_data_preparation.py

Cleans the raw simulated dataset and prepares it for modeling:
  1. Remove duplicate rows.
  2. Handle missing values (categorical -> mode imputation, numeric -> median).
  3. One-hot encode categorical features.
  4. Split into train (80%) / test (20%) sets.
  5. Persist cleaned_logistics_data.csv, train_data.csv, test_data.csv.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw_logistics_data.csv")

df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 1. Remove duplicates
# ---------------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "order_id"])
print(f"Removed {before - len(df)} duplicate rows")

# ---------------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------------
print("\nMissing values before imputation:")
print(df.isna().sum()[df.isna().sum() > 0])

df["weather"] = df["weather"].fillna(df["weather"].mode()[0])
df["rider_rating"] = df["rider_rating"].fillna(df["rider_rating"].median())

assert df.isna().sum().sum() == 0, "Missing values remain after imputation"

# ---------------------------------------------------------------------------
# 3. Feature engineering: encode categoricals
# ---------------------------------------------------------------------------
categorical_cols = [
    "traffic_level",
    "weather",
    "time_of_day",
    "day_of_week",
    "area_type",
    "vehicle_type",
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop identifier column (not a predictive feature)
df_encoded = df_encoded.drop(columns=["order_id"])

cleaned_path = os.path.join(DATA_DIR, "cleaned_logistics_data.csv")
df_encoded.to_csv(cleaned_path, index=False)
print(f"\nCleaned + encoded dataset saved to: {cleaned_path}")
print(f"Cleaned shape: {df_encoded.shape}")

# ---------------------------------------------------------------------------
# 4. Train / test split (80/20)
# ---------------------------------------------------------------------------
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

train_path = os.path.join(DATA_DIR, "train_data.csv")
test_path = os.path.join(DATA_DIR, "test_data.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nTrain set: {train_df.shape} -> {train_path}")
print(f"Test set : {test_df.shape} -> {test_path}")
