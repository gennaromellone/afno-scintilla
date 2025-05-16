import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AFNODataset(Dataset):
    def __init__(self, sim_dir, obs_dir, preload=False):
        self.sim_files = sorted(f for f in os.listdir(sim_dir) if f.startswith("simulated_") and f.endswith(".npz"))
        self.obs_files = sorted(f for f in os.listdir(obs_dir) if f.startswith("obs_") and f.endswith(".npz"))
        assert len(self.sim_files) == len(self.obs_files), "Mismatch between simulation and observation files"

        self.sim_dir = sim_dir
        self.obs_dir = obs_dir
        self.preload = preload

        if preload:
            self.sim_data = [np.load(os.path.join(sim_dir, f)) for f in self.sim_files]
            self.obs_data = [np.load(os.path.join(obs_dir, f)) for f in self.obs_files]
        else:
            self.sim_data = None
            self.obs_data = None

    def __len__(self):
        return len(self.sim_files)

    def __getitem__(self, idx):
        if self.preload:
            sim_npz = self.sim_data[idx]
            obs_npz = self.obs_data[idx]
        else:
            sim_npz = np.load(os.path.join(self.sim_dir, self.sim_files[idx]))
            obs_npz = np.load(os.path.join(self.obs_dir, self.obs_files[idx]))

        x = torch.from_numpy(sim_npz['input']).float()
        y = torch.from_numpy(sim_npz['target']).float()
        o = torch.from_numpy(obs_npz['obs']).float()
        return x, y, o