import os
import numpy as np
from tqdm import tqdm

# Directory degli .npz
obs_dir = "/storage/external_01/scintilla/processed_afno/training_2017/obs"
sim_dir = "/storage/external_01/scintilla/processed_afno/training_2017/sim"

# Funzione di analisi

def analyze_npz_dir(directory, key):
    files = sorted([f for f in os.listdir(directory) if f.endswith(".npz")])
    values = []
    per_file_stats = []

    print(f"\n📦 Found {len(files)} .npz files in: {directory}")

    for f in tqdm(files):
        path = os.path.join(directory, f)
        try:
            arr = np.load(path)[key]
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            values.append(arr)
            per_file_stats.append({
                "file": f,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            })
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")

    if values:
        all_data = np.concatenate(values)
        global_mean = all_data.mean()
        global_std = all_data.std()

        print("\n✅ Global stats across all non-NaN values:")
        print(f"Mean: {global_mean:.4f}")
        print(f"Std:  {global_std:.4f}")

        print("\n🔍 Top 5 most divergent files (by mean):")
        top_div = sorted(per_file_stats, key=lambda x: abs(x["mean"]))[-5:]
        for stat in top_div:
            print(f"{stat['file']} → mean: {stat['mean']:.4f}, std: {stat['std']:.4f}")
    else:
        print("⚠️  No usable data found in .npz files.")

# Analizza osservazioni
analyze_npz_dir(obs_dir, key="obs")

# Analizza simulazioni target
analyze_npz_dir(sim_dir, key="target")