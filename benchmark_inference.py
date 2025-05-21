import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.afno_modulus import AFNOModel, AFNODataset

# === Configurazione ===
CONFIG_PATH = "configs/config.yaml"
CHECKPOINT_PATH = "experiment01/best_model.pt"

# === Carica configurazione ===
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# === Dataset solo 1 esempio ===
dataset = AFNODataset(
    sim_dir=cfg["data"]["training_simulated_path"],
    obs_dir=cfg["data"]["interpolated_observation_path"],
    norm_file=cfg["data"].get("normalization_file"),
    time_window=cfg["model"]["time_window"]
)

# Carichiamo 1 solo sample per test
sample = dataset[0]
input_tensor = sample["input"].unsqueeze(0)  # Add batch dimension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modello ===
model = AFNOModel(
    img_shape=dataset.target_shape[1:],
    in_channels=dataset.input_shape[0],
    out_channels=dataset.target_shape[0],
    patch_size=cfg["model"].get("patch_size", [1, 1]),
    embed_dim=cfg["model"].get("embed_dim", 64),
    depth=cfg["model"].get("depth", 4),
    num_blocks=cfg["model"].get("num_blocks", 8),
    sparsity_threshold=cfg["model"].get("sparsity_threshold", 0.01),
    hard_thresholding_fraction=cfg["model"].get("hard_thresholding_fraction", 1.0)
).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# === Inference Benchmark ===
input_tensor = input_tensor.to(device)
n_repeats = 50
times = []

with torch.no_grad():
    for _ in range(n_repeats):
        start = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.time()
        times.append(end - start)

mean_time = np.mean(times)
print(f"\n⏱️ Inference time per 1-hour prediction (avg over {n_repeats} runs): {mean_time*1000:.2f} ms")
