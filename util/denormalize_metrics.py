import pandas as pd
import yaml
import argparse

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="training_metrics.csv", help="CSV file with normalized metrics")
parser.add_argument("--norm_file", type=str, required=True, help="YAML normalization file")
parser.add_argument("--field", type=str, required=True, help="Field name to denormalize against (e.g. PM10)")
args = parser.parse_args()

# === Load metrics ===
df = pd.read_csv(args.csv)

# === Load normalization stats ===
with open(args.norm_file) as f:
    stats = yaml.safe_load(f)

if args.field not in stats:
    raise ValueError(f"Field '{args.field}' not found in normalization file")

std = stats[args.field]["std"]

# === Denormalize RMSE and MAE ===
df["train_rmse_real"] = df["train_rmse"] * std
df["val_rmse_real"] = df["val_rmse"] * std
df["train_mae_real"] = df["train_mae"] * std
df["val_mae_real"] = df["val_mae"] * std

out_csv = args.csv.replace(".csv", f"_denorm_{args.field}.csv")
df.to_csv(out_csv, index=False)
print(f"✅ Denormalized metrics saved to {out_csv}")
