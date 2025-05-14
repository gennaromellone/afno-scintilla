import xarray as xr
import os
import glob

# Path ai tuoi file
chunks_folder = "data/processed/tmp_chunks"  # es. /data/afno/chunks/
merged_zarr = "data/processed/out.zarr"  # es. /data/afno/merged.zarr

print("=== Samples nei singoli chunks ===")
chunks = sorted(glob.glob(os.path.join(chunks_folder, "*.zarr")))

total_chunk_samples = 0
for chunk in chunks:
    ds = xr.open_zarr(chunk)
    n = ds['input'].shape[0]
    print(f"{os.path.basename(chunk)} → {n} samples")
    total_chunk_samples += n

print(f"\nTotale samples nei chunks: {total_chunk_samples}")

print("\n=== Samples nel merged finale ===")
ds_merged = xr.open_zarr(merged_zarr)
print(f"Samples nel merged: {ds_merged['input'].shape[0]}")

# Se hai un dataset pyTorch che fa il match:
print("\n=== Samples dopo matching osservazioni ===")

import os
import sys
import yaml
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

print(f"Samples finali dopo matching: {len(dataset)}")


