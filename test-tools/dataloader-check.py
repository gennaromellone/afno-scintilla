import xarray as xr

def inspect_zarr(zarr_path):
    print(f"🔍 Loading Zarr dataset from: {zarr_path}")
    
    ds = xr.open_zarr(zarr_path, chunks=None)

    if "target" not in ds:
        print("❌ 'target' variable not found in the dataset!")
        return

    target_shape = ds["target"].shape  # (N, C, H, W)

    if len(target_shape) != 4:
        print(f"❌ Unexpected target shape: {target_shape}")
        return

    out_channels = target_shape[1]
    print(f"✅ Detected target shape: {target_shape}")
    print(f"🎯 Number of output channels (out_channels): {out_channels}")

if __name__ == "__main__":
    # 🔁 Inserisci qui il percorso corretto
    zarr_path = "/storage/external_01/scintilla/processed_afno/training_2017.zarr"
    inspect_zarr(zarr_path)
