import os
import numpy as np
import yaml

def load_config():
    base_path = "/home/gmellone/afno-scintilla/configs"
    data_config_path = os.path.join(base_path, "data.yaml")
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg["training_simulated_path"], data_cfg['normalization_file']

def validate_npz_files(npz_dir):
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz") and f.startswith("simulated_"))
    if not files:
        print("❌ No .npz files found in directory.")
        return

    print(f"🔍 Found {len(files)} simulated .npz files")

    input_shapes = []
    target_shapes = []
    timestamps = set()
    errors = 0

    for f in files:
        try:
            data = np.load(os.path.join(npz_dir, f))
            input_arr = data['input']
            target_arr = data['target']
            ts = data['timestamp']

            input_shapes.append(input_arr.shape)
            target_shapes.append(target_arr.shape)

            if isinstance(ts, np.ndarray):
                ts = ts.item()  # safely extract scalar from array
            timestamps.add(str(ts))

            if np.isnan(input_arr).any():
                print(f"⚠️  NaNs found in input of {f}")
            if np.isnan(target_arr).any():
                print(f"⚠️  NaNs found in target of {f}")

        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            errors += 1

    unique_input_shapes = set(input_shapes)
    unique_target_shapes = set(target_shapes)

    print(f"✅ Unique input shapes: {unique_input_shapes}")
    print(f"✅ Unique target shapes: {unique_target_shapes}")
    print(f"✅ Total unique timestamps: {len(timestamps)}")
    print(f"✅ Completed with {errors} error(s)")

    # Compare with dataset_info.txt
    info_path = os.path.join(npz_dir, "dataset_info.txt")
    if os.path.exists(info_path):
        print("\n📄 Checking dataset_info.txt:")
        with open(info_path) as f:
            info_lines = f.readlines()
        info_dict = dict(line.strip().split(": ", 1) for line in info_lines if ": " in line)

        if "num_samples" in info_dict:
            declared_samples = int(info_dict["num_samples"])
            if declared_samples != len(files):
                print(f"❌ Mismatch in sample count: info={declared_samples} vs found={len(files)}")
            else:
                print("✅ Sample count matches dataset_info.txt")

        if "input_shape" in info_dict:
            if str(next(iter(unique_input_shapes))) != info_dict["input_shape"]:
                print(f"❌ Mismatch in input_shape: info={info_dict['input_shape']} vs actual={unique_input_shapes}")
            else:
                print("✅ input_shape matches dataset_info.txt")

        if "target_shape" in info_dict:
            if str(next(iter(unique_target_shapes))) != info_dict["target_shape"]:
                print(f"❌ Mismatch in target_shape: info={info_dict['target_shape']} vs actual={unique_target_shapes}")
            else:
                print("✅ target_shape matches dataset_info.txt")

    else:
        print("⚠️  dataset_info.txt not found for comparison")

def validate_normalization_file(norm_path):
    if not os.path.exists(norm_path):
        print("⚠️  Normalization file not found.")
        return

    with open(norm_path) as f:
        stats = yaml.safe_load(f)

    print("\n📊 Normalization Statistics:")
    for section in stats:
        mean = stats[section].get("mean")
        std = stats[section].get("std")
        print(f"  {section}: mean={mean}, std={std}")

    if any(std == 0 for section in stats for std in [stats[section].get("std")]):
        print("❌ Invalid std=0 in normalization file!")
    else:
        print("✅ Normalization stats look valid")

if __name__ == "__main__":
    data_dir, norm_file = load_config()
    validate_npz_files(data_dir)
    validate_normalization_file(norm_file)
