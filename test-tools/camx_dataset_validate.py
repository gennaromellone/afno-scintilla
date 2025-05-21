import os
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Config ===
base_path = "/home/gmellone/afno-scintilla/configs"
config_path = os.path.join(base_path, "config.yaml")

with open(config_path) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg['data']
sim_dir = data_cfg["training_simulated_path"]
norm_file = data_cfg.get("normalization_file")

# === Validation ===
logger.info(f"🔍 Validating simulated .npz files in: {sim_dir}")
files = sorted([f for f in os.listdir(sim_dir) if f.startswith("simulated_") and f.endswith(".npz")])

if not files:
    logger.error("❌ No .npz simulation files found.")
    exit(1)

logger.info(f"🔍 Found {len(files)} simulated .npz files")

bad_files = []
sample_shapes = []

for f in tqdm(files):
    try:
        data = np.load(os.path.join(sim_dir, f))
        input_data = data["input"]
        target_data = data["target"]
        ts = data["timestamp"]

        if np.isnan(input_data).any() or np.isnan(target_data).any():
            nan_ratio = (np.isnan(target_data).sum() / target_data.size) * 100
            if nan_ratio == 100.0:
                logger.warning(f"⚠️  High NaN ratio in {f} ({nan_ratio:.1f}%)")
            bad_files.append(f)

        sample_shapes.append((input_data.shape, target_data.shape))

    except Exception as e:
        logger.error(f"❌ Error reading {f}: {e}")

logger.info(f"✅ Validation complete: {len(files) - len(bad_files)} OK, {len(bad_files)} with issues")

# === Normalization stats ===
if norm_file and os.path.exists(norm_file):
    logger.info(f"📦 Validating normalization file: {norm_file}")
    with open(norm_file) as f:
        norm_stats = yaml.safe_load(f)

    for section in ["input", "target"]:
        if section not in norm_stats:
            logger.error(f"❌ Missing '{section}' section in normalization file")
            continue

        for k, v in norm_stats[section].items():
            mean = v.get("mean")
            std = v.get("std")
            if std is None or std == 0:
                logger.error(f"❌ Invalid std for {section}:{k} → {std}")
            else:
                logger.info(f"✅ {section}:{k} → mean={mean:.2f}, std={std:.2f}")
else:
    logger.warning("⚠️  Normalization file not found or not specified")
