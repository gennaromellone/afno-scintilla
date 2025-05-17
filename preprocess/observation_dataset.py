import os
import numpy as np
import pandas as pd
import yaml
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Config ===
base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path = os.path.join(base_path, "data.yaml")
model_config_path = os.path.join(base_path, "model.yaml")

with open(model_config_path) as f:
    model_cfg = yaml.safe_load(f)

with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)

obs_dir = data_cfg["obs_dir"]
out_dir = os.path.splitext(data_cfg["interpolated_observation_path"])[0] + "_npz"
os.makedirs(out_dir, exist_ok=True)

# === Get grid shape ===
img_shape = tuple(model_cfg["img_shape"])
output_vars = data_cfg["output_vars"]

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

# === Load CAMx timestamps from training_simulated_path ===
zarr_ts = []
ts_source = data_cfg.get("training_simulated_path")
if os.path.exists(ts_source):
    zarr_ts = sorted([f for f in os.listdir(ts_source) if f.endswith(".npz") and f.startswith("simulated_")])
    zarr_ts = [np.load(os.path.join(ts_source, f))['timestamp'] for f in zarr_ts]
    zarr_ts = [t.item() if isinstance(t, np.ndarray) else t for t in zarr_ts]

logger.info(f"Interpolating {len(zarr_ts)} timestamps from simulated dataset")

all_obs = []
valid_ts = []

for i, ts in enumerate(tqdm(zarr_ts)):
    daily_interp = []
    for var in output_vars:
        if var not in obs_data:
            daily_interp.append(np.full(img_shape, np.nan))
            continue

        df_ts = obs_data[var][obs_data[var]["timestamp"] == ts]

        if df_ts.empty:
            logger.warning(f"No data for {var} at {ts}, using NaN")
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
            daily_interp.append(grid)

    stacked = np.stack(daily_interp).astype(np.float32)
    all_obs.append(stacked)
    valid_ts.append(ts)

# === Compute normalization stats ===
parser = argparse.ArgumentParser()
parser.add_argument("--normalize", action="store_true", help="Apply normalization and save stats")
args = parser.parse_args()

norm_path = data_cfg['normalization_file']
norm_stats = {}
if args.normalize:
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_stats = yaml.safe_load(f)
        mean = norm_stats["target"]["mean"]
        std = norm_stats["target"]["std"]
        logger.info(f"Using normalization from simulated data: mean={mean:.4f}, std={std:.4f}")
        full_array = np.stack(all_obs)
        full_array = (full_array - mean) / std
        all_obs = [full_array[i] for i in range(len(full_array))]
    else:
        logger.warning("Normalization file not found; skipping normalization.")

# === Save per-sample npz ===
for i, (ts, arr) in enumerate(zip(valid_ts, all_obs)):
    fname = os.path.join(out_dir, f"obs_{i:06d}.npz")
    np.savez_compressed(fname, obs=arr, timestamp=ts)

logger.info(f"✅ Saved {len(all_obs)} observation files to {out_dir}")
