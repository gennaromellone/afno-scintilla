import os
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
import shutil
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path  = os.path.join(base_path, "data.yaml")
model_config_path  = os.path.join(base_path, "model.yaml")

with open(model_config_path) as f:
    model_cfg = yaml.safe_load(f)

with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)

output_path = data_cfg['training_simulated_path']

def expand_vars(var_list, species_vars):
    new_vars = []
    for var in var_list:
        if var == "PM10":
            new_vars += species_vars["PM25"] + species_vars["PM10_extra"]
        elif var == "PM25":
            new_vars += species_vars["PM25"]
        else:
            new_vars.append(var)
    return new_vars

def get_stack(ds_obj, var_list):
    return np.stack([ds_obj[var].isel(LAY=0).values for var in var_list], axis=1)

def process_file(index, files, camx_dir, out_dir, input_vars, output_vars, time_window, forecast_horizon):
    fname = files[index]
    date_str = fname.split("_")[1][:8]
    zarr_path = os.path.join(out_dir, f"chunk_{date_str}.zarr")

    if os.path.exists(zarr_path):
        logger.info(f"Skipping {date_str} (already exists)")
        return

    ds = xr.open_dataset(os.path.join(camx_dir, fname))

    if ds.sizes['TSTEP'] < time_window:
        logger.warning(f"Skipping {date_str} → insufficient timesteps")
        return

    prev_ds = None
    if index > 0:
        try:
            prev_ds = xr.open_dataset(os.path.join(camx_dir, files[index - 1]))
        except Exception:
            logger.warning(f"Prev day {files[index-1]} not found or broken")

    input_stack = get_stack(ds, input_vars)
    target_stack = get_stack(ds, output_vars)

    logger.info(f"Input stack shape: {input_stack.shape} → (T={input_stack.shape[0]}, V={input_stack.shape[1]}, H={input_stack.shape[2]}, W={input_stack.shape[3]})")
    logger.info(f"Target stack shape: {target_stack.shape} → (T={target_stack.shape[0]}, V={target_stack.shape[1]}, H={target_stack.shape[2]}, W={target_stack.shape[3]})")

    if prev_ds is not None:
        try:
            prev_input = get_stack(prev_ds, input_vars)[-time_window + 1:]
            prev_target = get_stack(prev_ds, output_vars)[-time_window + 1:]
            input_stack = np.concatenate([prev_input, input_stack], axis=0)
            target_stack = np.concatenate([prev_target, target_stack], axis=0)
        except Exception as e:
            logger.warning(f"Could not append previous data: {e}")

    input_data, target_data, ts = [], [], []
    base_time = datetime.strptime(f"{date_str}_00", "%Y%m%d_%H")

    for t in range(time_window - 1, len(input_stack) - forecast_horizon + 1):
        input_window = input_stack[t - time_window + 1: t + 1]
        target_time = t + forecast_horizon

        if target_time >= len(target_stack):
            continue

        target_frame = target_stack[target_time]

        if np.isnan(input_window).any() or np.isnan(target_frame).any():
            print("NaN value!")
            continue

        T, V, H, W = input_window.shape
        input_reshaped = input_window.reshape(T * V, H, W)

        input_data.append(input_reshaped)
        target_data.append(target_frame)

        true_time = base_time + timedelta(hours=target_time)
        ts.append(true_time.strftime("%Y%m%dT%H"))

    if not input_data or not target_data:
        logger.warning(f"Skipping {date_str} → no valid samples")
        return

    min_len = min(len(input_data), len(target_data))
    input_data = input_data[:min_len]
    target_data = target_data[:min_len]
    ts = ts[:min_len]

    dataset = xr.Dataset({
        "input": (("time", "input_channel", "y", "x"), np.array(input_data)),
        "target": (("time", "target_channel", "y", "x"), np.array(target_data)),
        "timestamps": ("time", np.array(ts, dtype="S32"))
    })

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    dataset = dataset.chunk({"time": -1})
    dataset.to_zarr(zarr_path, mode="w", consolidated=False)

    logger.info(f"✅ Saved {zarr_path} with {min_len} samples")

def build_all_chunks():

    camx_dir = data_cfg["camx_dir"]
    out_dir = data_cfg["training_checkpoints_dir"]
    os.makedirs(out_dir, exist_ok=True)

    input_vars = expand_vars(data_cfg["input_vars"], data_cfg["species_vars"])
    output_vars = expand_vars(data_cfg["output_vars"], data_cfg["species_vars"])

    files = sorted([f for f in os.listdir(camx_dir) if f.endswith(f"{model_cfg['grid']}.nc")])

    Parallel(n_jobs=4)(
        delayed(process_file)(
            i, files, camx_dir, out_dir,
            input_vars, output_vars,
            model_cfg["time_window"], model_cfg["forecast_horizon"]
        ) for i in tqdm(range(len(files)))
    )

if __name__ == "__main__":
    build_all_chunks()
