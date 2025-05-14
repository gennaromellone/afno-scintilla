import xarray as xr
import numpy as np

ds = xr.open_zarr("data/out.zarr", consolidated=False)

print("ZARR FINALE")
print("input shape:", ds["input"].shape)
print("target shape:", ds["target"].shape)

print("input mean:", np.nanmean(ds["input"].values))
print("target mean:", np.nanmean(ds["target"].values))

print("NaN in input:", np.isnan(ds["input"].values).sum())
print("NaN in target:", np.isnan(ds["target"].values).sum())

print("Sample timestamp:", ds["timestamps"].values[0].view('S32').astype(str))
