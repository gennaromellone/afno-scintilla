import os
import numpy as np
import argparse

def validate_obs_npz(obs_dir, simulated_dir):
    obs_files = sorted(f for f in os.listdir(obs_dir) if f.endswith(".npz") and f.startswith("obs_"))
    sim_files = sorted(f for f in os.listdir(simulated_dir) if f.endswith(".npz") and f.startswith("simulated_"))

    print(f"🔍 Found {len(obs_files)} observation files")
    print(f"🔍 Found {len(sim_files)} simulated files")

    if len(obs_files) != len(sim_files):
        print(f"❌ Mismatch in file count: {len(obs_files)} obs vs {len(sim_files)} simulated")
    else:
        print(f"✅ File count matches")

    errors = 0
    for i, (obs_name, sim_name) in enumerate(zip(obs_files, sim_files)):
        obs_path = os.path.join(obs_dir, obs_name)
        sim_path = os.path.join(simulated_dir, sim_name)
        try:
            obs = np.load(obs_path)
            sim = np.load(sim_path)

            obs_ts = obs["timestamp"]
            sim_ts = sim["timestamp"]
            obs_ts = obs_ts.decode() if isinstance(obs_ts, bytes) else str(obs_ts)
            sim_ts = sim_ts.decode() if isinstance(sim_ts, bytes) else str(sim_ts)

            if obs_ts != sim_ts:
                print(f"❌ Timestamp mismatch at index {i}: obs={obs_ts} vs sim={sim_ts}")
                errors += 1
                continue

            if "obs" not in obs:
                print(f"❌ Missing 'obs' key in {obs_name}")
                errors += 1
                continue

            obs_shape = obs["obs"].shape
            sim_target_shape = sim["target"].shape

            if obs_shape[1:] != sim_target_shape[1:]:
                print(f"❌ Spatial shape mismatch at index {i}: obs={obs_shape[1:]} vs sim={sim_target_shape[1:]}")
                errors += 1

            if np.isnan(obs["obs"]).mean() > 0.5:
                print(f"⚠️  High NaN ratio in {obs_name} ({np.isnan(obs['obs']).mean():.1%})")

        except Exception as e:
            print(f"❌ Error reading files at index {i}: {e}")
            errors += 1

    if errors == 0:
        print("✅ All observation files passed validation!")
    else:
        print(f"⚠️ Validation completed with {errors} issue(s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dir", required=True, help="Directory with obs_*.npz files")
    parser.add_argument("--sim_dir", required=True, help="Directory with simulated_*.npz files")
    args = parser.parse_args()

    validate_obs_npz(args.obs_dir, args.sim_dir)
