import os
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Load config ===
base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path = os.path.join(base_path, "data.yaml")
model_config_path = os.path.join(base_path, "model.yaml")

with open(model_config_path) as f:
    model_cfg = yaml.safe_load(f)

with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)

out_dir = data_cfg["training_simulated_path"]
os.makedirs(out_dir, exist_ok=True)

dataset_info = {
    "num_samples": 0,
    "input_shape": None,
    "target_shape": None,
    "input_vars": [],
    "target_vars": [],
    "grid_size": None,
    "time_window": model_cfg["time_window"],
    "forecast_horizon": model_cfg["forecast_horizon"]
}

def expand_vars(var_list, species_vars, expand_pm10=True):
    new_vars = []
    for var in var_list:
        if var == "PM10" and expand_pm10:
            new_vars += species_vars["PM25"] + species_vars["PM10_extra"]
        elif var == "PM25":
            new_vars += species_vars["PM25"]
        else:
            new_vars.append(var)
    return new_vars

def get_stack(ds_obj, var_list, species_vars):
    result = []
    for var in var_list:
        if var == "PM10":
            species = species_vars["PM25"] + species_vars["PM10_extra"]
            stack = np.stack([ds_obj[v].isel(LAY=0).values for v in species], axis=1)
            result.append(np.sum(stack, axis=1, keepdims=True))
        elif var == "PM25":
            species = species_vars["PM25"]
            stack = np.stack([ds_obj[v].isel(LAY=0).values for v in species], axis=1)
            result.append(np.sum(stack, axis=1, keepdims=True))
        else:
            arr = ds_obj[var].isel(LAY=0).values[:, None, :, :]
            result.append(arr)
    return np.concatenate(result, axis=1)

def compute_pm10(target_frame, expanded_output_vars, species_vars):
    pm10_species = species_vars["PM25"] + species_vars["PM10_extra"]
    indices = [i for i, var in enumerate(expanded_output_vars) if var in pm10_species]
    return target_frame[indices].sum(axis=0, keepdims=True)

def process_file(index, files, camx_dir, input_vars_raw, output_vars, time_window, forecast_horizon, start_sample_idx, single_target_pm10, expanded_output_vars, species_vars):
    fname = files[index]
    date_str = fname.split("_")[1][:8]
    ds_path = os.path.join(camx_dir, fname)

    ds = xr.open_dataset(ds_path)
    if ds.sizes['TSTEP'] < time_window:
        logger.warning(f"Skipping {date_str} → insufficient timesteps")
        return 0

    prev_ds = None
    if index > 0:
        try:
            prev_ds = xr.open_dataset(os.path.join(camx_dir, files[index - 1]))
        except Exception:
            logger.warning(f"Prev day {files[index - 1]} not found or broken")

    input_stack = get_stack(ds, input_vars_raw, species_vars)
    target_stack = get_stack(ds, expanded_output_vars, species_vars)

    if prev_ds is not None:
        try:
            prev_input = get_stack(prev_ds, input_vars_raw, species_vars)[-time_window + 1:]
            prev_target = get_stack(prev_ds, expanded_output_vars, species_vars)[-time_window + 1:]
            input_stack = np.concatenate([prev_input, input_stack], axis=0)
            target_stack = np.concatenate([prev_target, target_stack], axis=0)
        except Exception as e:
            logger.warning(f"Could not append previous data: {e}")

    base_time = datetime.strptime(f"{date_str}_00", "%Y%m%d_%H")
    sample_idx = 0

    for t in range(time_window - 1, len(input_stack) - forecast_horizon + 1):
        input_window = input_stack[t - time_window + 1: t + 1]
        target_time = t + forecast_horizon
        if target_time >= len(target_stack):
            continue

        target_frame = target_stack[target_time]
        if np.isnan(input_window).any() or np.isnan(target_frame).any():
            continue

        T, V, H, W = input_window.shape
        input_reshaped = input_window.reshape(T * V, H, W)

        if single_target_pm10:
            target_frame = compute_pm10(target_frame, expanded_output_vars, species_vars)

        timestamp = (base_time + timedelta(hours=target_time)).strftime("%Y%m%dT%H")
        out_fname = os.path.join(out_dir, f"simulated_{start_sample_idx + sample_idx:06d}.npz")
        np.savez_compressed(out_fname, input=input_reshaped.astype(np.float32), target=target_frame.astype(np.float32), timestamp=timestamp)

        if dataset_info["input_shape"] is None:
            dataset_info["input_shape"] = input_reshaped.shape
            dataset_info["target_shape"] = target_frame.shape
            dataset_info["grid_size"] = (H, W)

        sample_idx += 1

    logger.info(f"✅ {date_str} → saved {sample_idx} samples")
    return sample_idx

def build_all_npz():
    camx_dir = data_cfg["camx_dir"]
    input_vars_raw = data_cfg["input_vars"]
    output_vars_raw = data_cfg["output_vars"]

    expand_input = input_vars_raw not in ("PM10", "PM25")
    input_vars = expand_vars(input_vars_raw, data_cfg["species_vars"], expand_pm10=expand_input)
    output_vars_expanded = expand_vars(output_vars_raw, data_cfg["species_vars"], expand_pm10=True)
    single_target_pm10 = output_vars_raw == ["PM10"]

    files = sorted([f for f in os.listdir(camx_dir) if f.endswith(f"{model_cfg['grid']}.nc")])

    dataset_info["input_vars"] = input_vars_raw
    dataset_info["target_vars"] = output_vars_expanded if not single_target_pm10 else ["PM10"]

    sample_counter = [0]

    def wrapped_process(i):
        sample_idx = sum(sample_counter)
        result = process_file(
            i, files, camx_dir,
            input_vars_raw, output_vars_raw,
            model_cfg["time_window"],
            model_cfg["forecast_horizon"],
            sample_idx,
            single_target_pm10,
            output_vars_expanded,
            data_cfg["species_vars"]
        )
        sample_counter[0] += result
        return result

    Parallel(n_jobs=1)(delayed(wrapped_process)(i) for i in tqdm(range(len(files))))
    dataset_info["num_samples"] = sample_counter[0]

    # === Save summary file ===
    info_file = os.path.join(out_dir, "dataset_info.txt")
    with open(info_file, "w") as f:
        for k, v in dataset_info.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"📍 Saved dataset info to {info_file}")

if __name__ == "__main__":
    build_all_npz()
