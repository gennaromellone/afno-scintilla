import os
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--cleanall", action="store_true", help="Delete all files with 100% NaN content or timestamp mismatch")
args = parser.parse_args()

# === Config ===
base_path = "/home/gmellone/afno-scintilla/configs"
config_path = os.path.join(base_path, "config.yaml")

with open(config_path) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg['data']
obs_dir = data_cfg["interpolated_observation_path"]
sim_dir = data_cfg["training_simulated_path"]
norm_file = data_cfg.get("normalization_file")

# === Load file maps by timestamp ===
def build_timestamp_map(directory, prefix):
    mapping = {}
    for f in os.listdir(directory):
        if f.startswith(prefix) and f.endswith(".npz"):
            path = os.path.join(directory, f)
            try:
                with np.load(path) as data:
                    ts = data["timestamp"].item()
                    mapping[ts] = f
            except Exception as e:
                logger.error(f"❌ Failed to read timestamp from {f}: {e}")
    return mapping

logger.info("🔍 Building timestamp maps for simulated and observed data")
sim_map = build_timestamp_map(sim_dir, "simulated_")
obs_map = build_timestamp_map(obs_dir, "obs_")

common_ts = sorted(set(sim_map.keys()) & set(obs_map.keys()))
logger.info(f"🔍 Found {len(sim_map)} simulated files")
logger.info(f"🔍 Found {len(obs_map)} observation files")
logger.info(f"🔗 Matched {len(common_ts)} timestamp-aligned pairs")

bad_files = []
mismatched_ts = []
sample_shapes = []

for ts in tqdm(common_ts):
    sim_file = sim_map[ts]
    obs_file = obs_map[ts]

    sim_path = os.path.join(sim_dir, sim_file)
    obs_path = os.path.join(obs_dir, obs_file)

    try:
        sim_npz = np.load(sim_path)
        obs_npz = np.load(obs_path)

        sim_data = sim_npz["target"]
        obs_data = obs_npz["obs"] if "obs" in obs_npz else obs_npz["target"]
        ts_sim = sim_npz["timestamp"]
        ts_obs = obs_npz["timestamp"]

        if ts_sim != ts_obs:
            logger.warning(f"⏱️  Timestamp mismatch: {sim_file} vs {obs_file} → {ts_sim} ≠ {ts_obs}")
            mismatched_ts.append((sim_file, obs_file))
            if args.cleanall:
                os.remove(sim_path)
                os.remove(obs_path)
                logger.info(f"🗑️  Removed mismatched files: {sim_file}, {obs_file}")
            continue

        if np.isnan(obs_data).any():
            nan_ratio = (np.isnan(obs_data).sum() / obs_data.size) * 100
            if nan_ratio == 100.0:
                logger.warning(f"⚠️  High NaN ratio in {obs_file} ({nan_ratio:.1f}%)")
                bad_files.append(obs_file)
                if args.cleanall:
                    os.remove(obs_path)
                    os.remove(sim_path)
                    logger.info(f"🗑️  Removed NaN files: {obs_file}, {sim_file}")

        sample_shapes.append(obs_data.shape)

    except Exception as e:
        logger.error(f"❌ Error reading pair {sim_file}, {obs_file}: {e}")

logger.info(f"✅ Validation complete: {len(common_ts) - len(bad_files) - len(mismatched_ts)} OK, {len(bad_files)} NaN, {len(mismatched_ts)} timestamp mismatches")

# === Normalization stats for observations (PM10 only) ===
if norm_file and os.path.exists(norm_file):
    logger.info(f"📦 Validating normalization file: {norm_file}")
    with open(norm_file) as f:
        norm_stats = yaml.safe_load(f)

    section = "target"
    key = "PM10"
    if section not in norm_stats:
        logger.error(f"❌ Missing '{section}' section in normalization file")
    elif key not in norm_stats[section]:
        logger.error(f"❌ Missing '{key}' in normalization file under '{section}'")
    else:
        mean = norm_stats[section][key].get("mean")
        std = norm_stats[section][key].get("std")
        if std is None or std == 0:
            logger.error(f"❌ Invalid std for {section}:{key} → {std}")
        else:
            logger.info(f"✅ {section}:{key} → mean={mean:.2f}, std={std:.2f}")
else:
    logger.warning("⚠️  Normalization file not found or not specified")
