import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

input_file = "pm10_days_forecasts_last.csv"
# Carica il file
df = pd.read_csv(input_file)

# Pulisci NaN
df_valid = df.dropna(subset=["pm10", "pm10_nearest_forecast"])

# Raggruppa per data
grouped = df_valid.groupby("date_start").agg({
    "pm10": "mean",
    "pm10_nearest_forecast": "mean"
}).reset_index()

# Estrai osservazioni e simulazioni mediati
y_true = grouped["pm10"].values
y_pred = grouped["pm10_nearest_forecast"].values

# Calcola metriche
mae_camx = mean_absolute_error(y_true, y_pred)
rmse_camx = np.sqrt(mean_squared_error(y_true, y_pred))
r2_camx = r2_score(y_true, y_pred)

print(f"MAE: {mae_camx:.2f} µg/m³")
print(f"RMSE: {rmse_camx:.2f} µg/m³")
print(f"R²: {r2_camx:.4f}")

config_text = f"Target Scores: {input_file}\n\nMAE: {mae_camx:.2f} µg/m³\nRMSE: {rmse_camx:.2f} µg/m³\nR²: {r2_camx:.4f}"
with open("target_scores.txt", "w") as f:
    f.write(config_text)
