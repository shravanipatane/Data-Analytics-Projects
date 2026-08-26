"""
05_hyperparameter_tuning.py

Applies 5-fold cross-validation and GridSearchCV to tune the Decision Tree
and Random Forest regressors, then compares the tuned Random Forest against
the baseline Linear Regression using cross-validated RMSE.

Outputs:
  - results/cross_validation_results.csv
  - visualizations/05_cross_validation_results.png
  - models/random_forest_model.pkl (overwritten with the tuned best estimator)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

TARGET = "delivery_time_minutes"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# 1. GridSearchCV — Decision Tree
# ---------------------------------------------------------------------------
dt_param_grid = {
    "max_depth": [4, 6, 8, 10, 12],
    "min_samples_leaf": [5, 10, 20, 30],
}
dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=kfold,
    n_jobs=-1,
)
dt_grid.fit(X_train, y_train)
print("Best Decision Tree params:", dt_grid.best_params_)
print(f"Best Decision Tree CV RMSE: {-dt_grid.best_score_:.3f}\n")

# ---------------------------------------------------------------------------
# 2. GridSearchCV — Random Forest
# ---------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16],
    "min_samples_leaf": [2, 5, 10],
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=kfold,
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print("Best Random Forest params:", rf_grid.best_params_)
print(f"Best Random Forest CV RMSE: {-rf_grid.best_score_:.3f}\n")

# Persist the tuned Random Forest as the "production" ensemble model
best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

best_dt = dt_grid.best_estimator_
joblib.dump(best_dt, os.path.join(MODEL_DIR, "decision_tree_model.pkl"))

# ---------------------------------------------------------------------------
# 3. Cross-validate all three models (tuned) for a fair comparison
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (tuned)": best_dt,
    "Random Forest (tuned)": best_rf,
}

cv_rows = []
cv_raw = {}
for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kfold, n_jobs=-1
    )
    rmse_scores = -scores
    cv_raw[name] = rmse_scores
    cv_rows.append(
        {
            "model": name,
            "mean_cv_rmse": rmse_scores.mean(),
            "std_cv_rmse": rmse_scores.std(),
            "fold_1": rmse_scores[0],
            "fold_2": rmse_scores[1],
            "fold_3": rmse_scores[2],
            "fold_4": rmse_scores[3],
            "fold_5": rmse_scores[4],
        }
    )
    print(f"{name:24s} | mean CV RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.3f})")

cv_df = pd.DataFrame(cv_rows)
cv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
cv_df.to_csv(cv_path, index=False)
print(f"\nCross-validation results saved to: {cv_path}")

# ---------------------------------------------------------------------------
# 4. Visualization: cross-validation RMSE spread (boxplot)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(8, 6))
plt.boxplot(
    [cv_raw[name] for name in models.keys()],
    labels=list(models.keys()),
    patch_artist=True,
    boxprops=dict(facecolor="#2E86AB", alpha=0.6),
    medianprops=dict(color="red"),
)
plt.ylabel("RMSE (minutes)")
plt.title("5-Fold Cross-Validation RMSE by Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "05_cross_validation_results.png"), dpi=150)
plt.close()

print("Cross-validation visualization saved to ../visualizations/05_cross_validation_results.png")
