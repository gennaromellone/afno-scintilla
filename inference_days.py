import os
import yaml
import torch
import numpy as np
import pandas as pd
import xarray as xr
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from models.afno_modulus import AFNOModel

# === PATHS ===
csv_path = "util/pm10_days_forecasts_last.csv"
sim_npz_dir = "/storage/external_01/scintilla/processed_afno/training_2017/sim_DEC/"  # npz con 'input'


camx_example = "/storage/external_01/scintilla/CAMx/DATA/camx_20171203.avrg.grd01.nc"
norm_path = "/home/gmellone/afno-scintilla/normalization_stats_combined.yaml"
config_path = "configs/config.yaml"
checkpoint_path = "experiment01/best_model.pt"

pred_npz_dir = "/storage/external_01/scintilla/processed_afno/training_2017/pred/"
os.makedirs(pred_npz_dir, exist_ok=True)

output_csv = "pm10_obs_forecast_with_afno.csv"
# === Configura il logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Load normalization stats ===
with open(norm_path) as f:
    stats = yaml.safe_load(f)
input_stats = stats["input"]
target_mean = stats["target"]["PM10"]["mean"]
target_std = stats["target"]["PM10"]["std"]

# === Load config ===
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# === Load CAMx grid and KDTree ===
ds = xr.open_dataset(camx_example)
lat = ds["latitude"].values
lon = ds["longitude"].values
H, W = lat.shape
coords = np.stack([lat.ravel(), lon.ravel()], axis=1)
tree = cKDTree(coords)

# === Load AFNO model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = np.load(os.path.join(sim_npz_dir, sorted(os.listdir(sim_npz_dir))[0]))["input"]
model = AFNOModel(
    img_shape=dummy_input.shape[1:],
    in_channels=dummy_input.shape[0],
    out_channels=1,
    patch_size=cfg["model"]["patch_size"],
    embed_dim=cfg["model"]["embed_dim"],
    depth=cfg["model"]["depth"],
    num_blocks=cfg["model"]["num_blocks"],
    sparsity_threshold=cfg["model"].get("sparsity_threshold", 0.01),
    hard_thresholding_fraction=cfg["model"].get("hard_thresholding_fraction", 1.0)
).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# === Load and filter CSV ===
df = pd.read_csv(csv_path, parse_dates=["date_start"])
df = df[df["date_start"].dt.month == 12].copy()
df = df[[
    "station", "date_start", "date_end", "pm10", "unit", "lat", "lon", "altitude", "period", "pm10_nearest_forecast"
]].copy()  # ensure only relevant columns are preserved

df["timestamp"] = df["date_start"].dt.strftime("%Y%m%d")
station_coords = df[["lat", "lon"]].drop_duplicates().values
station_ids = df["station"].unique()
station_kd_idx = {}
for latlon in station_coords:
    _, idx = tree.query(latlon)
    station_kd_idx[(round(latlon[0], 4), round(latlon[1], 4))] = np.unravel_index(idx, (H, W))

# === Build daily predictions ===
df["pm10_afno_prediction"] = np.nan
for date in tqdm(df["date_start"].unique(), desc="Daily Inference"):
    day_preds = {}
    for _, row in df[df["date_start"] == date].iterrows():
        key = (round(row["lat"], 4), round(row["lon"], 4))
        day_preds.setdefault(key, [])

    for h in range(24):
        ts = (pd.to_datetime(date) + timedelta(hours=h)).strftime("%Y%m%dT%H")
        npz_path = os.path.join(sim_npz_dir, f"simulated_{ts}.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"Missing file: {npz_path}")
            continue

        d = np.load(npz_path)
        x = d["input"].astype(np.float32)
        for i in range(x.shape[0]):
            ch = f"ch_{i}"
            if ch in input_stats:
                x[i] = (x[i] - input_stats[ch]["mean"]) / input_stats[ch]["std"]

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).squeeze(0).cpu().numpy()[0]

        # Save hourly prediction to .npz
        pred_path = os.path.join(pred_npz_dir, f"afno_pred_{ts}.npz")
        np.savez_compressed(pred_path, pred=pred.astype(np.float32), timestamp=ts)

        for coord, (i, j) in station_kd_idx.items():
            pred_val = pred[i, j] * target_std + target_mean
            if coord in day_preds:
                day_preds[coord].append(pred_val)

    for coord, values in day_preds.items():
        if len(values) >= 12:
            mean_val = np.mean(values)
            mask = (
                (df["date_start"] == pd.to_datetime(date)) &
                (df["lat"].round(4) == coord[0]) & (df["lon"].round(4) == coord[1])
            )
            df.loc[mask, "pm10_afno_prediction"] = mean_val
            logger.info(f"{date.date()} - {coord} - Pred: {mean_val:.2f} µg/m³")
        else:
            logger.warning(f"{date.date()} - {coord} - Incomplete hours: {len(values)}")

# === Save results ===
df.to_csv(output_csv, index=False)
logger.info(f"Saved predictions to {output_csv}")
print(f"✅ Saved predictions to {output_csv}")
