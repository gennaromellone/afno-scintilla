
"""
Interpolate air quality observations onto CAMx model grid for AFNO target generation.

This script loads hourly air quality observations (e.g., PM10, PM2.5, NO2) from CSV files
and interpolates them onto the CAMx model grid, matching the spatial resolution of CAMx outputs.

Output: npz files (one per timestamp) aligned with simulated CAMx training samples.

Usage:
    python preprocess/interpolate_obs_to_grid.py

Author:
    Gennaro Mellone, 2025
"""
import os
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import griddata
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Load config ===
base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path = os.path.join(base_path, "data.yaml")
model_config_path = os.path.join(base_path, "model.yaml")

with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)
with open(model_config_path) as f:
    model_cfg = yaml.safe_load(f)

obs_dir = data_cfg["obs_dir"]
out_dir = data_cfg["interpolated_observation_path"]
sim_npz_dir = data_cfg["training_simulated_path"]
os.makedirs(out_dir, exist_ok=True)

output_vars = data_cfg["output_vars"]
img_shape = tuple(model_cfg["img_shape"])

# === Load observations CSV ===
obs_data = {}
for var in output_vars:
    var_file = os.path.join(obs_dir, f"{var}_hour.csv")
    if not os.path.exists(var_file):
        logger.warning(f"File not found: {var_file}")
        continue

    df = pd.read_csv(var_file, header=None)
    df.columns = ["id", "start", "end", "value", "unit", "lat", "lon", "alt", "freq"]
    df["start"] = pd.to_datetime(df["start"])
    df["timestamp"] = df["start"].dt.strftime("%Y%m%dT%H")
    obs_data[var] = df
    logger.info(f"Loaded {len(df)} rows for {var}")

# === Extract timestamps from simulated .npz ===
sim_files = sorted([f for f in os.listdir(sim_npz_dir) if f.endswith(".npz") and f.startswith("simulated_")])
logger.info(f"Found {len(sim_files)} simulated npz files")

for i, fname in enumerate(tqdm(sim_files)):
    sim_path = os.path.join(sim_npz_dir, fname)
    try:
        sim_data = np.load(sim_path)
        ts = sim_data["timestamp"]
        ts = ts.decode() if isinstance(ts, bytes) else str(ts)
    except Exception as e:
        logger.warning(f"Failed to read {fname}: {e}")
        continue

    daily_interp = []
    for var in output_vars:
        if var not in obs_data:
            daily_interp.append(np.full(img_shape, np.nan))
            continue

        df_ts = obs_data[var][obs_data[var]["timestamp"] == ts]

        if df_ts.empty:
            logger.warning(f"No data for {var} at {ts}, using fallback")
            daily_interp.append(np.full(img_shape, np.nan))
        else:
            grid = griddata(
                df_ts[["lat", "lon"]].values,
                df_ts["value"].values,
                (np.linspace(df_ts["lat"].min(), df_ts["lat"].max(), img_shape[0])[:, None],
                 np.linspace(df_ts["lon"].min(), df_ts["lon"].max(), img_shape[1])[None, :]),
                method='nearest',
                fill_value=np.nan
            )
            nan_ratio = np.isnan(grid).mean()
            if nan_ratio > 0:
                logger.warning(f"{ts} → interpolated grid has {nan_ratio:.1%} NaNs")
            daily_interp.append(grid)

    stacked = np.stack(daily_interp).astype(np.float32)  # shape (C, H, W)
    out_path = os.path.join(out_dir, f"obs_{i:06d}.npz")
    np.savez_compressed(out_path, obs=stacked, timestamp=ts)

logger.info(f"✅ Saved {len(sim_files)} observation .npz files to {out_dir}")
