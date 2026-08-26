"""
04_model_evaluation.py

Evaluates the three trained models on the held-out test set using:
  - RMSE  (Root Mean Squared Error)
  - MAE   (Mean Absolute Error)
  - R^2   (Coefficient of Determination)

Generates and saves:
  - results/model_performance.csv
  - results/feature_importance.csv
  - data/prediction_results.csv (best model predictions vs actual)
  - visualizations/01_actual_vs_predicted.png
  - visualizations/02_model_comparison.png
  - visualizations/03_feature_importance.png
  - visualizations/04_residual_analysis.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

TARGET = "delivery_time_minutes"
plt.style.use("seaborn-v0_8-whitegrid")

test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

model_files = {
    "Linear Regression": "linear_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
}

performance_rows = []
predictions = {}

for name, fname in model_files.items():
    model = joblib.load(os.path.join(MODEL_DIR, fname))
    preds = model.predict(X_test)
    predictions[name] = preds

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    performance_rows.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"{name:20s} | RMSE: {rmse:6.3f} | MAE: {mae:6.3f} | R2: {r2:.4f}")

perf_df = pd.DataFrame(performance_rows).sort_values("RMSE")
perf_path = os.path.join(RESULTS_DIR, "model_performance.csv")
perf_df.to_csv(perf_path, index=False)
print(f"\nModel performance table saved to: {perf_path}")

best_model_name = perf_df.iloc[0]["model"]
print(f"Best model (lowest RMSE): {best_model_name}")

# ---------------------------------------------------------------------------
# Save prediction_results.csv for the best model
# ---------------------------------------------------------------------------
pred_df = X_test.copy()
pred_df["actual_delivery_time_minutes"] = y_test.values
pred_df["predicted_delivery_time_minutes"] = np.round(predictions[best_model_name], 2)
pred_df["absolute_error"] = np.round(
    np.abs(pred_df["actual_delivery_time_minutes"] - pred_df["predicted_delivery_time_minutes"]), 2
)
pred_out_path = os.path.join(DATA_DIR, "prediction_results.csv")
pred_df.to_csv(pred_out_path, index=False)
print(f"Prediction results saved to: {pred_out_path}")

# ---------------------------------------------------------------------------
# Visualization 1: Actual vs Predicted (best model)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions[best_model_name], alpha=0.35, s=18, color="#2E86AB")
lims = [min(y_test.min(), predictions[best_model_name].min()),
        max(y_test.max(), predictions[best_model_name].max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title(f"Actual vs Predicted Delivery Time — {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "01_actual_vs_predicted.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Visualization 2: Model comparison bar chart (RMSE, MAE, R2)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#2E86AB", "#A23B72", "#F18F01"]
for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
    ax.bar(perf_df["model"], perf_df[metric], color=colors)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Model Performance Comparison")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "02_model_comparison.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Visualization 3: Feature importance (Random Forest)
# ---------------------------------------------------------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)
importances = importances.sort_values(ascending=False).head(12)

importances.to_frame("importance").reset_index().rename(
    columns={"index": "feature"}
).to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="#2E86AB")
plt.xlabel("Importance")
plt.title("Top 12 Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "03_feature_importance.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Visualization 4: Residual analysis (best model)
# ---------------------------------------------------------------------------
residuals = y_test.values - predictions[best_model_name]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(predictions[best_model_name], residuals, alpha=0.35, s=18, color="#A23B72")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Delivery Time (minutes)")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=30, color="#F18F01", edgecolor="black")
axes[1].set_xlabel("Residual (minutes)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")
fig.suptitle(f"Residual Analysis — {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "04_residual_analysis.png"), dpi=150)
plt.close()

print("\nVisualizations saved to ../visualizations/")
