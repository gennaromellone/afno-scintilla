import os
import xarray as xr
import logging
import shutil
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def merge_chunks(chunks_dir, out_path):
    chunk_paths = sorted([os.path.join(chunks_dir, d) for d in os.listdir(chunks_dir) if d.endswith(".zarr")])

    if not chunk_paths:
        raise RuntimeError("No chunks found!")

    if os.path.exists(out_path):
        logger.info(f"Removing existing {out_path}")
        shutil.rmtree(out_path)

    for i, path in enumerate(chunk_paths):
        try:
            ds = xr.open_zarr(path, consolidated=False)
        except Exception as e:
            raise RuntimeError(f"Error reading {path}: {e}")

        logger.info(f"{'Writing' if i == 0 else 'Appending'} {os.path.basename(path)} with {ds.sizes['time']} samples")

        if i == 0:
            # Primo chunk → scrittura normale
            ds.to_zarr(out_path, mode="w", consolidated=False)
        else:
            # Chunk successivi → append su dimensione time
            ds.to_zarr(out_path, mode="a", append_dim="time", consolidated=False)

    logger.info(f"✅ Final dataset saved to {out_path}")


if __name__ == "__main__":
    base_path = "/home/gmellone/afno-scintilla/configs"
    data_config_path  = os.path.join(base_path, "data.yaml")

    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    chunks_dir = data_cfg['training_checkpoints_dir']
    out_path = data_cfg['training_simulated_path']
    merge_chunks(chunks_dir, out_path)
