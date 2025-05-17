import os
import numpy as np
import yaml
import argparse

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--autoclean", action="store_true", help="Automatically delete files with 100% NaNs")
args = parser.parse_args()

# === Load config ===
base_path = "/home/gmellone/afno-scintilla/configs"
data_config_path = os.path.join(base_path, "data.yaml")
with open(data_config_path) as f:
    data_cfg = yaml.safe_load(f)

obs_npz_dir = data_cfg["interpolated_observation_path"]
sim_npz_dir = data_cfg["training_simulated_path"]

obs_files = sorted(f for f in os.listdir(obs_npz_dir) if f.endswith(".npz") and f.startswith("obs_"))
sim_files = sorted(f for f in os.listdir(sim_npz_dir) if f.endswith(".npz") and f.startswith("simulated_"))
print(f"🔍 Found {len(obs_files)} observation files")
print(f"🔍 Found {len(sim_files)} simulated files")

errors = 0
nan_warnings = 0
shapes = set()
obs_timestamps = set()

deleted_files = 0

for fname in obs_files:
    path = os.path.join(obs_npz_dir, fname)
    try:
        data = np.load(path)
        obs = data['obs']
        ts = data['timestamp']
        if isinstance(ts, np.ndarray):
            ts = ts.item()
        obs_timestamps.add(str(ts))

        shapes.add(obs.shape)

        nan_ratio = np.isnan(obs).mean()
        if nan_ratio > 0.2:
            print(f"⚠️  High NaN ratio in {fname} ({nan_ratio:.1%})")
            nan_warnings += 1

        if args.autoclean and nan_ratio == 1.0:
            os.remove(path)
            sim_path = os.path.join(sim_npz_dir, fname.replace("obs_", "simulated_"))
            if os.path.exists(sim_path):
                os.remove(sim_path)
            print(f"🧹 Deleted {fname} and matching simulation file")
            deleted_files += 1

    except Exception as e:
        print(f"❌ Error reading {fname}: {e}")
        errors += 1

print(f"✅ Unique shapes: {shapes}")
print(f"✅ Total obs timestamps: {len(obs_timestamps)}")
print(f"⚠️  Files with high NaNs: {nan_warnings}")
print(f"❌ Total read errors: {errors}")
if args.autoclean:
    print(f"🧹 Deleted files with 100% NaNs: {deleted_files}")

# === Check normalization file consistency ===
norm_path = os.path.join(data_cfg["training_simulated_path"], "normalization_stats.yaml")
if os.path.exists(norm_path):
    with open(norm_path) as f:
        norm_stats = yaml.safe_load(f)

    mean = norm_stats["target"].get("mean")
    std = norm_stats["target"].get("std")
    print("\n📊 Normalization stats from simulation:")
    print(f"  mean = {mean}, std = {std}")
    if std == 0:
        print("❌ Invalid normalization: std = 0")
    else:
        print("✅ Normalization looks valid")
else:
    print("⚠️  Normalization file not found")

# === Compare with simulation timestamps ===
sim_timestamps = set()
for f in sim_files:
    try:
        data = np.load(os.path.join(sim_npz_dir, f))
        ts = data['timestamp']
        if isinstance(ts, np.ndarray):
            ts = ts.item()
        sim_timestamps.add(str(ts))
    except Exception as e:
        print(f"❌ Error reading simulated file {f}: {e}")

missing_in_obs = sim_timestamps - obs_timestamps
missing_in_sim = obs_timestamps - sim_timestamps

if not missing_in_obs and not missing_in_sim:
    print("✅ Timestamps match exactly between observations and simulations")
else:
    if missing_in_obs:
        print(f"❌ {len(missing_in_obs)} timestamps missing in obs (e.g. {list(missing_in_obs)[:5]})")
    if missing_in_sim:
        print(f"❌ {len(missing_in_sim)} timestamps missing in sim (e.g. {list(missing_in_sim)[:5]})")
