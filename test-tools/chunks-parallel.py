import os
import xarray as xr
import numpy as np

chunks_dir = "data/processed/tmp_chunks"

chunks = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".zarr")])
print(f"Trovati {len(chunks)} chunks")

for chunk in chunks[:3]:  # controllo primi 3
    ds = xr.open_zarr(os.path.join(chunks_dir, chunk), consolidated=False)
    print(f"\nChunk {chunk}")
    print("input shape:", ds["input"].shape)
    print("target shape:", ds["target"].shape)

    print("input mean:", np.nanmean(ds["input"].values))
    print("input min:", np.nanmin(ds["input"].values))
    print("input max:", np.nanmax(ds["input"].values))

    print("target mean:", np.nanmean(ds["target"].values))
    print("target min:", np.nanmin(ds["target"].values))
    print("target max:", np.nanmax(ds["target"].values))
