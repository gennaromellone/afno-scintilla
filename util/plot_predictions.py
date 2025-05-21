import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "pm10_obs_forecast_with_afno.csv"
output_plot = "monthly_pm10_comparison.png"

# === Load data ===
df = pd.read_csv(csv_path, parse_dates=["date_start"])

# === Filter for December and valid AFNO predictions ===
df = df[df["date_start"].dt.month == 12]
df = df[df["pm10_afno_prediction"].notna()]

# === Daily means across stations ===
daily_means = df.groupby("date_start")[["pm10", "pm10_nearest_forecast", "pm10_afno_prediction"]].mean()

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(daily_means.index, daily_means["pm10"], label="Observed PM10", marker='o')
plt.plot(daily_means.index, daily_means["pm10_nearest_forecast"], label="CAMx Forecast", marker='x')
plt.plot(daily_means.index, daily_means["pm10_afno_prediction"], label="AFNO Prediction", marker='s')
plt.xlabel("Date")
plt.ylabel("PM10 (µg/m³)")
plt.title("Daily Mean PM10 — Observed vs CAMx vs AFNO")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_plot)
plt.show()

print(f"✅ Plot saved to {output_plot}")
