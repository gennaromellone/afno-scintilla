import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse

# === Load config ===
config_path = os.path.join("/home/gmellone/afno-scintilla/configs", "config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

def load_timestamp(path):
    try:
        return np.load(path)["timestamp"].item()
    except Exception:
        return None

def validate_dataset(sim_dir, obs_dir, norm_file=None, autoclean=False):
    sim_files = sorted([f for f in os.listdir(sim_dir) if f.startswith("simulated_") and f.endswith(".npz")])
    obs_files = sorted([f for f in os.listdir(obs_dir) if f.startswith("obs_") and f.endswith(".npz")])

    sim_map = {}
    obs_map = {}

    for f in sim_files:
        path = os.path.join(sim_dir, f)
        ts = load_timestamp(path)
        if ts:
            sim_map[ts] = f

    for f in obs_files:
        path = os.path.join(obs_dir, f)
        ts = load_timestamp(path)
        if ts:
            obs_map[ts] = f

    common_ts = sorted(set(sim_map) & set(obs_map))
    missing_sim = sorted(set(obs_map) - set(sim_map))
    missing_obs = sorted(set(sim_map) - set(obs_map))

    print(f"\n📊 Dataset alignment check")
    print(f"✅ Found {len(common_ts)} aligned files")
    if missing_sim:
        print(f"❌ Missing {len(missing_sim)} simulation files")
        for ts in missing_sim:
            print(f"  - Missing sim for timestamp: {ts}")
            if autoclean:
                obs_path = os.path.join(obs_dir, obs_map[ts])
                if os.path.exists(obs_path):
                    os.remove(obs_path)

    if missing_obs:
        print(f"❌ Missing {len(missing_obs)} observation files")
        for ts in missing_obs:
            print(f"  - Missing obs for timestamp: {ts}")
            if autoclean:
                sim_path = os.path.join(sim_dir, sim_map[ts])
                if os.path.exists(sim_path):
                    os.remove(sim_path)

    # Check for NaNs in observation files
    nan_obs = []
    for ts in tqdm(common_ts, desc="🔍 Checking for NaNs in obs"):
        obs_path = os.path.join(obs_dir, obs_map[ts])
        data = np.load(obs_path)["obs"]
        if np.isnan(data).mean() > 0.99:
            nan_obs.append(ts)
            if autoclean:
                os.remove(obs_path)
                sim_path = os.path.join(sim_dir, sim_map[ts])
                if os.path.exists(sim_path):
                    os.remove(sim_path)

    print(f"\n⚠️ Found {len(nan_obs)} observation files with >99% NaNs")
    if autoclean:
        print(f"🧹 Auto-cleaned {len(nan_obs)} files from both sim and obs folders")

    # Optional: Check normalization file
    if norm_file and os.path.exists(norm_file):
        with open(norm_file) as f:
            stats = yaml.safe_load(f)
        print("\n📦 Normalization file loaded:")
        for key in stats:
            print(f"- {key}: {list(stats[key].keys())}")
    else:
        print("\n📭 No normalization file found or specified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoclean", action="store_true", help="Automatically remove files with >99% NaNs or unmatched pairs")
    args = parser.parse_args()

    sim_dir = cfg["data"]["training_simulated_path"]
    obs_dir = cfg["data"]["interpolated_observation_path"]
    norm_file = cfg["data"].get("normalization_file")

    validate_dataset(sim_dir, obs_dir, norm_file, autoclean=args.autoclean)