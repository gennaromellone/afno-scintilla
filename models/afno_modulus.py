import os
import numpy as np
import torch
import pandas as pd
import yaml
import torch.nn as nn

from tqdm import tqdm
from physicsnemo.models.afno import AFNO
from torch.utils.data import Dataset, DataLoader, random_split

# Model AFNO
class AFNOModel(nn.Module):
    def __init__(self, img_shape, in_channels, out_channels=1,
                 patch_size=(1, 1), embed_dim=64, depth=4,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super().__init__()

        self.model = AFNO(
            inp_shape=list(img_shape),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )

    def forward(self, x):
        return self.model(x)

# Dataset AFNO
class AFNODataset(Dataset):
    def __init__(self, sim_dir, obs_dir, norm_file=None, time_window=6):
        self.sim_dir = sim_dir
        self.obs_dir = obs_dir
        self.time_window = time_window

        self.sim_map = self._build_map(sim_dir, prefix="simulated_")
        self.obs_map = self._build_map(obs_dir, prefix="obs_")

        self.timestamps = sorted(list(set(self.sim_map.keys()) & set(self.obs_map.keys())))

        if not self.timestamps:
            raise RuntimeError("No aligned simulation and observation timestamps found.")

        self.input_shape = None
        self.target_shape = None

        print("[INFO] Data assumed to be pre-normalized. Skipping normalization during loading.")

    def _build_map(self, directory, prefix):
        mapping = {}
        for f in os.listdir(directory):
            if f.startswith(prefix) and f.endswith(".npz"):
                try:
                    data = np.load(os.path.join(directory, f))
                    ts = data["timestamp"].item()
                    mapping[ts] = f
                except Exception:
                    continue
        return mapping

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        sim_path = os.path.join(self.sim_dir, self.sim_map[ts])
        obs_path = os.path.join(self.obs_dir, self.obs_map[ts])

        sim_data = np.load(sim_path)
        obs_data = np.load(obs_path)

        input = sim_data["input"].astype(np.float32)
        target_sim = sim_data["target"].astype(np.float32)
        target_obs = obs_data["obs"].astype(np.float32)

        if self.input_shape is None:
            self.input_shape = input.shape
            self.target_shape = target_sim.shape
            print(f"[DEBUG] Sample input mean: {input.mean():.4f}, std: {input.std():.4f}")
            print(f"[DEBUG] Sample target_sim mean: {target_sim.mean():.4f}, std: {target_sim.std():.4f}")
            print(f"[DEBUG] Sample target_obs mean: {target_obs.mean():.4f}, std: {target_obs.std():.4f}")

        return {
            "input": torch.from_numpy(input),
            "target_sim": torch.from_numpy(target_sim),
            "target_obs": torch.from_numpy(target_obs),
            "timestamp": ts
        }

# Training and Validation AFNO
class AFNOTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Dataset ===
        full_dataset = AFNODataset(
            sim_dir=cfg["data"]["training_simulated_path"],
            obs_dir=cfg["data"]["interpolated_observation_path"],
            norm_file=cfg["data"].get("normalization_file"),
            time_window=cfg["model"]["time_window"]
        )
        if full_dataset.input_shape is None or full_dataset.target_shape is None:
            _ = full_dataset[0]

        val_ratio = cfg["training"].get("validation_set", 0.2)
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg["training"]["batch_size"],
                                       shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)

        self.val_loader = DataLoader(self.val_dataset, batch_size=cfg["training"]["batch_size"],
                                     shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=True)

        # === Model ===
        in_ch = full_dataset.input_shape[0]
        out_ch = full_dataset.target_shape[0]
        img_shape = full_dataset.target_shape[1:]

        self.model = AFNOModel(
            img_shape=img_shape,
            in_channels=in_ch,
            out_channels=out_ch,
            patch_size=cfg["model"].get("patch_size", [1, 1]),
            embed_dim=cfg["model"].get("embed_dim", 64),
            depth=cfg["model"].get("depth", 4),
            num_blocks=cfg["model"].get("num_blocks", 8),
            sparsity_threshold=cfg["model"].get("sparsity_threshold", 0.01),
            hard_thresholding_fraction=cfg["model"].get("hard_thresholding_fraction", 1.0)
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["training"]["lr"])
        self.criterion = nn.MSELoss()
        self.alpha = cfg["training"].get("alpha", 1.0)
        self.beta = cfg["training"].get("beta", 1.0)
        self.early_stopping_patience = cfg["training"].get("early_stopping", 10)

        self.experiment_folder = cfg["training"].get("experiment_folder", "experiment01")
        if not os.path.exists(self.experiment_folder):
            print("Creating dir:", self.experiment_folder)
            os.makedirs(self.experiment_folder)

        self.log_path = os.path.join(self.experiment_folder, 'training_metrics.csv')
        self.checkpoint_path = os.path.join(self.experiment_folder, 'best_model.pt')

        self.best_val_loss = float("inf")
        self.early_counter = 0

        self.metric_log = []

    def compute_metrics(self, pred, target):
        mse = self.criterion(pred, target).item()
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        return mse, rmse, mae

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_rmse, total_mae = 0, 0, 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            x = batch["input"].to(self.device)
            y_sim = batch["target_sim"].to(self.device)
            y_obs = batch["target_obs"].to(self.device)

            y_pred = self.model(x)
            loss_sim = self.criterion(y_pred, y_sim)
            loss_obs = self.criterion(y_pred, y_obs)
            loss = self.alpha * loss_sim + self.beta * loss_obs

            loss.backward()
            self.optimizer.step()

            _, rmse, mae = self.compute_metrics(y_pred, y_obs)
            total_loss += loss.item()
            total_rmse += rmse
            total_mae += mae

        n = len(self.train_loader)
        return total_loss / n, total_rmse / n, total_mae / n

    def validate(self):
        self.model.eval()
        total_loss, total_rmse, total_mae = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                x = batch["input"].to(self.device)
                y_sim = batch["target_sim"].to(self.device)
                y_obs = batch["target_obs"].to(self.device)

                y_pred = self.model(x)
                loss_sim = self.criterion(y_pred, y_sim)
                loss_obs = self.criterion(y_pred, y_obs)
                loss = self.alpha * loss_sim + self.beta * loss_obs

                _, rmse, mae = self.compute_metrics(y_pred, y_obs)
                total_loss += loss.item()
                total_rmse += rmse
                total_mae += mae

        n = len(self.val_loader)
        return total_loss / n, total_rmse / n, total_mae / n

    def train(self):
        num_epochs = self.cfg["training"]["epochs"]
        print("\n🚀 Starting training on", self.device)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_rmse, train_mae = self.train_one_epoch()
            val_loss, val_rmse, val_mae = self.validate()

            print(f"\n📈 Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            self.metric_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae
            })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("✅ Best model saved")
            else:
                self.early_counter += 1
                if self.early_counter >= self.early_stopping_patience:
                    print("🛑 Early stopping triggered")
                    break

        pd.DataFrame(self.metric_log).to_csv(self.log_path, index=False)
        print(f"📁 Training log saved to {self.log_path}")
