import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd
import random

from models.afno_modulus import AFNODataset, AFNOModel

# === Load config ===
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# === Load and subset dataset ===
full_dataset = AFNODataset(
    sim_dir=cfg["data"]["training_simulated_path"],
    obs_dir=cfg["data"]["interpolated_observation_path"],
    norm_file=cfg["data"]["normalization_file"],
    time_window=cfg["model"]["time_window"]
)

_ = full_dataset[0]  # Force init shapes
subset_indices = random.sample(range(len(full_dataset)), min(300, len(full_dataset)))
train_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                          shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)

# === Setup model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_ch = full_dataset.input_shape[0]
out_ch = full_dataset.target_shape[0]
img_shape = full_dataset.target_shape[1:]

model = AFNOModel(
    img_shape=img_shape,
    in_channels=in_ch,
    out_channels=out_ch,
    patch_size=cfg["model"].get("patch_size", [1, 1]),
    embed_dim=cfg["model"].get("embed_dim", 64),
    depth=cfg["model"].get("depth", 4),
    num_blocks=cfg["model"].get("num_blocks", 8),
    sparsity_threshold=cfg["model"].get("sparsity_threshold", 0.01),
    hard_thresholding_fraction=cfg["model"].get("hard_thresholding_fraction", 1.0)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
criterion = nn.MSELoss()

alpha = cfg["training"].get("alpha", 1.0)
beta = cfg["training"].get("beta", 1.0)

# === Log ===
log = []

print("\n🚀 Starting training for 10 epochs on 300 samples")

for epoch in range(1, 11):
    model.train()
    total_loss, total_sim, total_obs = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        optimizer.zero_grad()
        x = batch["input"].to(device)
        y_sim = batch["target_sim"].to(device)
        y_obs = batch["target_obs"].to(device)

        y_pred = model(x)
        loss_sim = criterion(y_pred, y_sim)
        loss_obs = criterion(y_pred, y_obs)
        loss = alpha * loss_sim + beta * loss_obs

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sim += loss_sim.item()
        total_obs += loss_obs.item()

    avg_loss = total_loss / len(train_loader)
    avg_sim = total_sim / len(train_loader)
    avg_obs = total_obs / len(train_loader)
    print(f"Epoch {epoch:02d} | Total Loss: {avg_loss:.4f} | Sim Loss: {avg_sim:.4f} | Obs Loss: {avg_obs:.4f}")

    log.append({
        "epoch": epoch,
        "loss_total": avg_loss,
        "loss_sim": avg_sim,
        "loss_obs": avg_obs
    })

pd.DataFrame(log).to_csv("training_temp_log.csv", index=False)
print("📁 Log saved to training_temp_log.csv")
