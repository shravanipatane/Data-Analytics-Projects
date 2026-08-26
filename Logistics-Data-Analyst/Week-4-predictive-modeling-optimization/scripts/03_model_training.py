"""
03_model_training.py

Trains three regression models to predict `delivery_time_minutes`:
  1. Linear Regression   - interpretable baseline, assumes linear relationships
  2. Decision Tree        - captures non-linear interactions, easy to explain
  3. Random Forest         - ensemble of trees, usually best accuracy/robustness

Each trained model is persisted with joblib into ../models/ for reuse in
evaluation, tuning, and the optimization stage.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "delivery_time_minutes"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

feature_names = list(X_train.columns)
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}

for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"  -> saved to {model_path}")
    print(f"  -> Train R^2: {train_score:.4f} | Test R^2: {test_score:.4f}\n")

print("All models trained and saved to ../models/")
