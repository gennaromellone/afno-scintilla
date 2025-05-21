import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Paths ===
preds_path = "predictions/afno_outputs.npz"  # your model predictions
norm_path = "configs/normalization_stats_combined.yaml"
out_csv = "eval_denormalized_scores.csv"

# === Load normalization stats ===
with open(norm_path) as f:
    stats = yaml.safe_load(f)

mean = stats["target"]["PM10"]["mean"]
std = stats["target"]["PM10"]["std"]

# === Load model outputs ===
data = np.load(preds_path)
y_pred = data["pred"]  # shape: (N, H, W)
y_true = data["true"]  # shape: (N, H, W)

# === Denormalize ===
y_pred_denorm = y_pred * std + mean
y_true_denorm = y_true * std + mean

# === Flatten for metric computation ===
y_pred_flat = y_pred_denorm.flatten()
y_true_flat = y_true_denorm.flatten()

mask = ~np.isnan(y_true_flat)
y_pred_flat = y_pred_flat[mask]
y_true_flat = y_true_flat[mask]

# === Compute metrics ===
mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
r2 = r2_score(y_true_flat, y_pred_flat)

print("\n🎯 Denormalized Evaluation Metrics:")
print(f"MAE:  {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R²:   {r2:.4f}")

# Save to CSV
pd.DataFrame([{"mae": mae, "rmse": rmse, "r2": r2}]).to_csv(out_csv, index=False)
print(f"\n📁 Saved results to {out_csv}")
