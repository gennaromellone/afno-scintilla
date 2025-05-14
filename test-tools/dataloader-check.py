from torch.utils.data import DataLoader
import yaml
import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_dataset.afno_dataset import AFNOZarrObsDatasetOptimized

with open("preprocess/config_vars.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = AFNOZarrObsDatasetOptimized(
    sim_zarr_path="data/processed/out.zarr",
    obs_npz_path="data/processed/afno_obs_interpolated.npz",
    input_vars=cfg["input_vars"],
    output_vars=cfg["output_vars"],
    species_vars=cfg["species_vars"]
)

print("Totale samples:", len(dataset))

x, y = dataset[0]

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("Input mean:", torch.nanmean(x).item())
print("Target mean:", torch.nanmean(y).item())
