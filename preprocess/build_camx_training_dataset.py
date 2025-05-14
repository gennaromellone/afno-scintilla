import os
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm
import logging
import shutil
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def expand_vars(var_list, species_vars, var_type="input"):
    new_vars = []
    for var in var_list:
        if var == "PM10":
            logger.info(f"Expanding PM10 in {var_type}_vars into species...")
            new_vars += species_vars["PM25"] + species_vars["PM10_extra"]
        elif var == "PM25":
            logger.info(f"Expanding PM25 in {var_type}_vars into species...")
            new_vars += species_vars["PM25"]
        else:
            new_vars.append(var)
    return new_vars


def process_file(fpath, date_str, input_vars_exp, output_vars_exp, time_window, forecast_horizon):
    ds = xr.open_dataset(fpath)

    input_stack = [ds[var].isel(LAY=0).values for var in input_vars_exp]
    input_stack = np.stack(input_stack, axis=1)

    target_stack = [ds[var].isel(LAY=0).values for var in output_vars_exp]
    target_stack = np.stack(target_stack, axis=1)

    input_data, target_data, timestamps = [], [], []

    for t in range(0, 24 - time_window - forecast_horizon + 1):
        input_window = input_stack[t:t + time_window]
        target_time = t + time_window + forecast_horizon - 1
        target_frame = target_stack[target_time]

        T, V, H, W = input_window.shape
        input_reshaped = input_window.reshape(T * V, H, W)

        input_data.append(input_reshaped)
        target_data.append(target_frame)
        timestamps.append(f"{date_str}_h{target_time:02d}")

    ts_chunk = np.array(timestamps).astype("S32").view("uint8").reshape(-1, 32)

    return np.array(input_data), np.array(target_data), ts_chunk



def build_dataset(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    grid = cfg["grid"]
    camx_dir = cfg["camx_dir"]
    time_window = cfg["time_window"]
    forecast_horizon = cfg["forecast_horizon"]
    input_vars = cfg["input_vars"]
    output_vars = cfg["output_vars"]
    species_vars = cfg["species_vars"]

    output_path = "data/processed/afno_simulation_data.zarr"

    input_vars_exp = expand_vars(input_vars, species_vars, "input")
    output_vars_exp = expand_vars(output_vars, species_vars, "output")

    files = sorted(f for f in os.listdir(camx_dir) if f.endswith(f".{grid}.nc"))
    logger.info(f"Found {len(files)} CAMx files.")

    if os.path.exists(output_path):
        logger.info(f"Deleting existing {output_path}")
        shutil.rmtree(output_path)

    for idx, fname in enumerate(tqdm(files)):
        date_str = fname.split("_")[1][:8]

        input_chunk, target_chunk, ts_chunk = process_file(
            os.path.join(camx_dir, fname),
            date_str,
            input_vars_exp,
            output_vars_exp,
            time_window,
            forecast_horizon
        )

        dataset = xr.Dataset({
            "input": (("time", "input_channel", "y", "x"), input_chunk),
            "target": (("time", "target_channel", "y", "x"), target_chunk),
            "timestamps": (("time", "string32"), ts_chunk)
        })


        if idx == 0:
            dataset.to_zarr(output_path, mode="w", consolidated=False)
        else:
            dataset.to_zarr(output_path, mode="a", append_dim="time", consolidated=False)

        logger.info(f"Appended {input_chunk.shape[0]} samples from {date_str}")

    logger.info(f"Dataset saved successfully to {output_path}")


if __name__ == "__main__":
    build_dataset("preprocess/config_vars.yaml")
