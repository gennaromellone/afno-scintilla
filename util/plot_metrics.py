import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# === Percorsi ===

metrics_csv_path = "/home/gmellone/afno-scintilla/experiment01/training_metrics.csv"
norm_stats_path = "/home/gmellone/afno-scintilla/normalization_stats_combined.yaml"
output_dir = "/home/gmellone/afno-scintilla/experiment01/plots"
os.makedirs(output_dir, exist_ok=True)

# === Carica le metriche ===
df = pd.read_csv(metrics_csv_path)

# === Carica lo std da normalizzazione ===
with open(norm_stats_path) as f:
    stats = yaml.safe_load(f)
std_pm10 = stats["target"]["PM10"]["std"]

# === Denormalizza le metriche ===
df["train_rmse_denorm"] = df["train_rmse"] * std_pm10
df["val_rmse_denorm"] = df["val_rmse"] * std_pm10
df["train_mae_denorm"] = df["train_mae"] * std_pm10
df["val_mae_denorm"] = df["val_mae"] * std_pm10

# === Funzione di plotting ===
def plot_metric(metric_name, ylabel, denorm=False):
    suffix = "_denorm" if denorm else ""
    train_col = f"train_{metric_name}{suffix}"
    val_col = f"val_{metric_name}{suffix}"

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df[train_col], label="Train", marker="o")
    plt.plot(df["epoch"], df[val_col], label="Validation", marker="s")
    plt.title(f"{metric_name.upper()} ({'Denormalized' if denorm else 'Normalized'})")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}{suffix}.png"))
    plt.close()

# === Plot ===
plot_metric("loss", "Loss")
plot_metric("rmse", "RMSE")
plot_metric("mae", "MAE")
plot_metric("rmse", "RMSE (µg/m³)", denorm=True)
plot_metric("mae", "MAE (µg/m³)", denorm=True)

# === Estrazione dell'epoca con migliore val_loss ===
best = df.loc[df["val_loss"].idxmin()]
mae = best["val_mae_denorm"]
rmse = best["val_rmse_denorm"]
epoch = int(best["epoch"])

# Calcolo R² (approssimato sul dataset validazione)
r2 = 1 - (rmse**2) / (df["val_rmse_denorm"].mean()**2 + 1e-6)

print("\n📊 Best Validation Metrics:")
print(f"Epoch: {epoch}")
print(f"MAE:   {mae:.2f} µg/m³")
print(f"RMSE:  {rmse:.2f} µg/m³")
print(f"R²:    {r2:.4f}")