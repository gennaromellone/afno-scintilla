import os
import torch
from torch.utils.data import DataLoader
from dataset.afno_dataset_v2 import AFNODataset
import argparse
from tqdm import tqdm
import yaml
import time
import subprocess


def load_config():
    base_path = "/home/gmellone/afno-scintilla/configs"
    data_config_path = os.path.join(base_path, "data.yaml")

    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    sim_dir = data_cfg["training_simulated_path"]
    obs_dir = data_cfg["interpolated_observation_path"]
    return sim_dir, obs_dir

def get_gpu_util():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"])
        return int(output.decode("utf-8").strip().split("\n")[0])
    except Exception:
        return -1

def test_afno_dataset(sim_dir, obs_dir, batch_size=4, num_workers=2, device="cuda"):
    dataset = AFNODataset(sim_dir, obs_dir, preload=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"\nTesting AFNODataset with {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}, Device: {device}")

    try:
        start = time.time()
        gpu_util_logs = []

        for batch in tqdm(loader, desc="Running test", unit="batch"):
            x, y, o = [b.to(device, non_blocking=True) for b in batch]
            _ = x + y + o  # simulate model input
            gpu_util = get_gpu_util()
            if gpu_util >= 0:
                gpu_util_logs.append(gpu_util)

        end = time.time()
        duration = end - start
        avg_gpu = sum(gpu_util_logs) / len(gpu_util_logs) if gpu_util_logs else 0

        print("\n✅ AFNODataset passed GPU test without errors.")
        print(f"⏱ Total time: {duration:.2f}s, Avg time per batch: {duration / len(loader):.4f}s")
        if gpu_util_logs:
            print(f"🚀 Avg GPU Utilization: {avg_gpu:.1f}% over {len(gpu_util_logs)} batches")
        else:
            print("⚠️ Could not read GPU utilization during test.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    sim_dir, obs_dir = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_afno_dataset(sim_dir, obs_dir, args.batch_size, args.num_workers, device)
