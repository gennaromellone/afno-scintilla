import os
import numpy as np
import argparse

def validate_npz_directory(npz_dir, check_nan=True):
    info_path = os.path.join(npz_dir, "dataset_info.txt")
    if not os.path.exists(info_path):
        raise RuntimeError("Missing dataset_info.txt")

    with open(info_path) as f:
        info = {}
        for line in f:
            k, v = line.strip().split(":", 1)
            info[k.strip()] = eval(v.strip())

    expected_input_shape = tuple(info["input_shape"])
    expected_target_shape = tuple(info["target_shape"])
    expected_num_samples = int(info["num_samples"])

    print(f"🔍 Expected input shape: {expected_input_shape}")
    print(f"🔍 Expected target shape: {expected_target_shape}")
    print(f"🔍 Expected number of samples: {expected_num_samples}")

    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz") and f.startswith("simulated_")])
    if len(files) != expected_num_samples:
        print(f"❌ Found {len(files)} .npz files, expected {expected_num_samples}")
    else:
        print(f"✅ Found {len(files)} .npz files")

    errors = 0
    for i, fname in enumerate(files):
        path = os.path.join(npz_dir, fname)
        try:
            data = np.load(path)
            if "input" not in data or "target" not in data or "timestamp" not in data:
                print(f"❌ {fname} missing required keys")
                errors += 1
                continue

            if data["input"].shape != expected_input_shape:
                print(f"❌ {fname} input shape mismatch: {data['input'].shape}")
                errors += 1

            if data["target"].shape != expected_target_shape:
                print(f"❌ {fname} target shape mismatch: {data['target'].shape}")
                errors += 1

            if check_nan and (np.isnan(data["input"]).any() or np.isnan(data["target"]).any()):
                print(f"⚠️  {fname} contains NaNs")
                errors += 1

        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
            errors += 1

        if i % 100 == 0:
            print(f"Progress: {i}/{len(files)}")

    if errors == 0:
        print("✅ All files passed validation!")
    else:
        print(f"⚠️ Validation completed with {errors} issue(s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_dir", help="Directory containing simulated_*.npz files")
    parser.add_argument("--no-nan-check", action="store_true", help="Skip NaN validation")
    args = parser.parse_args()

    validate_npz_directory(args.npz_dir, check_nan=not args.no_nan_check)
