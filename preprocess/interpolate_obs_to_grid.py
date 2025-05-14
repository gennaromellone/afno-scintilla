import os
import numpy as np
import pandas as pd
import yaml
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
import logging
from datetime import datetime, timedelta


"""
Interpolate air quality observations onto CAMx model grid for AFNO target generation.

This script loads hourly air quality observations (e.g., PM10, PM2.5, NO2) from CSV files 
and interpolates them onto the CAMx model grid, matching the spatial resolution of CAMx outputs. 

Output: npz file with interpolated observations for AFNO target generation.

Usage:
------
python preprocess/interpolate_obs_to_grid.py

Author:
-------
Developed by Gennaro Mellone, 2025
"""


# === Setup logger ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path  = os.path.join(base_path, "data.yaml")
model_config_path  = os.path.join(base_path, "model.yaml")

with open(model_config_path) as f:
    model_cfg = yaml.safe_load(f)

with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)

output_path = data_cfg['interpolated_observation_path']

def interpolate_to_grid(df, shape, method="nearest"):
    """Interpolate observations over grid of given shape."""
    valid = df.dropna(subset=["lat", "lon", "value"])
    if valid.empty:
        return np.full(shape, np.nan)

    points = valid[["lat", "lon"]].values
    values = valid["value"].values

    lat_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), shape[0])
    lon_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), shape[1])
    grid_lat, grid_lon = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)

    interpolated = griddata(points, values, grid_coords, method=method, fill_value=np.nan)
    return interpolated.reshape(shape)


def interpolate_obs_to_camx_grid():

    img_shape = tuple(model_cfg["img_shape"])
    output_vars = data_cfg["output_vars"]
    grid_id = model_cfg.get("grid", "grd01")

    logger.info(f"Target grid shape: {img_shape}")
    logger.info(f"Output vars: {output_vars}")

    # === Load observations ===
    obs_data = {}
    for var in output_vars:
        var_file = os.path.join(data_cfg["obs_dir"], f"{var}_hour.csv")
        if not os.path.exists(var_file):
            logger.warning(f"File not found: {var_file}")
            continue

        df = pd.read_csv(var_file, header=None)
        df.columns = ["id", "start", "end", "value", "unit", "lat", "lon", "alt", "freq"]
        df["start"] = pd.to_datetime(df["start"])
        df["timestamp"] = df["start"].dt.strftime("%Y%m%d%H")
        obs_data[var] = df
        logger.info(f"Loaded {len(df)} rows for {var}")

    # === Process CAMx files ===
    camx_files = sorted(f for f in os.listdir(data_cfg["camx_dir"]) if grid_id in f and f.endswith(".nc"))
    logger.info(f"Found {len(camx_files)} CAMx files")

    obs_target = []
    timestamps = []

    for fname in tqdm(camx_files):
        date_str = fname.split("_")[1][:8]
        ds = xr.open_dataset(os.path.join(data_cfg["camx_dir"], fname))
        hours = ds.sizes.get("TSTEP", 24)
        base_time = datetime.strptime(f"{date_str}_00", "%Y%m%d_%H")

        #logger.info(f"Processing CAMx file: {fname} ({hours} hours)")

        for h in range(hours):
            ts_key = f"{date_str}{h:02d}"
            daily_interp = []

            for var in output_vars:
                if var not in obs_data:
                    daily_interp.append(np.full(img_shape, np.nan))
                    continue

                df_ts = obs_data[var][obs_data[var]["timestamp"] == ts_key]

                if df_ts.empty:
                    logger.warning(f"No data for {var} at {ts_key}, using 0.0 fallback")
                    daily_interp.append(np.full(img_shape, np.nan))
                else:
                    grid = interpolate_to_grid(df_ts, img_shape, 'nearest')
                    nan_ratio = np.isnan(grid).mean()
                    if nan_ratio > 0:
                        logger.warning(f"Interpolated grid at {ts_key} for {var} contains {nan_ratio:.2%} NaN values")
                    daily_interp.append(grid)

            obs_target.append(np.stack(daily_interp))

            true_time = base_time + timedelta(hours=h)
            timestamps.append(true_time.strftime("%Y%m%dT%H"))

    logger.info("Saving output...")
    np.savez(output_path,
             obs_target=np.array(obs_target),
             timestamps=np.array(timestamps))

    logger.info(f"Interpolation completed successfully. Output saved to {output_path}")


if __name__ == "__main__":
    interpolate_obs_to_camx_grid()
