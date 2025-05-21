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

from normalization import load_normalization_stats, apply_normalization

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Config ===
base_path = "/home/gmellone/afno-scintilla/configs"
config_path = os.path.join(base_path, "config.yaml")

with open(config_path) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg['data']
model_cfg = cfg['model']

obs_dir = data_cfg["obs_dir"]
camx_dir = data_cfg["camx_dir"]
out_dir = data_cfg["interpolated_observation_path"]
norm_out_path = os.path.join(out_dir, "normalization_stats_obs.yaml")
os.makedirs(out_dir, exist_ok=True)

observation_accumulator = {}

def interpolate_to_grid(df, shape, method="nearest"):
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

def update_stats(accumulator, array, names):
    for i, name in enumerate(names):
        ch_data = array[i].flatten()
        ch_data = ch_data[~np.isnan(ch_data)]  # remove NaNs
        if name not in accumulator:
            accumulator[name] = []
        accumulator[name].append(ch_data)

def finalize_stats(accumulator):
    stats = {}
    for name, values in accumulator.items():
        arr = np.concatenate(values)
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }
    return stats

def process_observation_files(normalize=False, collect_stats=False):
    img_shape = tuple(model_cfg["img_shape"])
    output_vars = data_cfg["output_vars"]
    grid_id = model_cfg.get("grid", "grd01")
    norm_file = data_cfg.get("normalization_file_obs") or data_cfg.get("normalization_file")

    logger.info(f"Target grid shape: {img_shape}")
    logger.info(f"Output vars: {output_vars}")

    norm_stats = None
    if normalize:
        norm_stats = load_normalization_stats(norm_file)
        if norm_stats is None or "target" not in norm_stats:
            logger.warning("⚠️  Normalization stats missing or incomplete – disabling normalization")
            normalize = False
        else:
            logger.info(f"📦 Applying normalization from: {norm_file}")
            for var in output_vars:
                if var in norm_stats["target"]:
                    m = norm_stats["target"][var]["mean"]
                    s = norm_stats["target"][var]["std"]
                    logger.info(f"→ {var}: mean={m:.2f}, std={s:.2f}")
                else:
                    logger.error(f"❌ Missing normalization stats for variable: {var}")
                    raise KeyError(f"Missing normalization stats for variable: {var}")

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

    camx_files = sorted(f for f in os.listdir(camx_dir) if grid_id in f and f.endswith(".nc"))
    logger.info(f"Found {len(camx_files)} CAMx files")

    idx = 0
    for fname in tqdm(camx_files):
        date_str = fname.split("_")[1][:8]
        ds = xr.open_dataset(os.path.join(camx_dir, fname))
        hours = ds.sizes.get("TSTEP", 24)
        base_time = datetime.strptime(f"{date_str}_00", "%Y%m%d_%H")

        for h in range(hours):
            ts_iso = f"{date_str}T{h:02d}"
            daily_interp = []

            for var in output_vars:
                if var not in obs_data:
                    daily_interp.append(np.full(img_shape, np.nan))
                    continue

                df_ts = obs_data[var][obs_data[var]["timestamp"] == ts_iso]

                if df_ts.empty:
                    daily_interp.append(np.full(img_shape, np.nan))
                else:
                    grid = interpolate_to_grid(df_ts, img_shape, 'nearest')
                    daily_interp.append(grid)

            obs_arr = np.stack(daily_interp)

            if collect_stats:
                update_stats(observation_accumulator, obs_arr, output_vars)

            if normalize:
                obs_arr = apply_normalization(
                    obs_arr,
                    stats=norm_stats["target"],
                    names=output_vars,
                    kind="target"
                )
                logger.info(f"🧪 After normalization → mean: {obs_arr.mean():.4f}, std: {obs_arr.std():.4f}")

            if not collect_stats:
                out_path = os.path.join(out_dir, f"obs_{ts_iso}.npz")
                if os.path.exists(out_path):
                    os.remove(out_path)
                np.savez_compressed(out_path, obs=obs_arr.astype(np.float32), timestamp=ts_iso)
                idx += 1

    if collect_stats:
        stats = {"target": finalize_stats(observation_accumulator)}
        with open(norm_out_path, "w") as f:
            yaml.dump(stats, f)
        logger.info(f"📦 Saved observation normalization stats to {norm_out_path}")
    else:
        logger.info(f"✅ Saved {idx} observation .npz files to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect_stats", "normalize_and_save"], required=True,
                        help="Mode: collect_stats or normalize_and_save")
    args = parser.parse_args()

    if args.mode == "collect_stats":
        process_observation_files(normalize=False, collect_stats=True)
    elif args.mode == "normalize_and_save":
        process_observation_files(normalize=True, collect_stats=False)

if __name__ == "__main__":
    main()