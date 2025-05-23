import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

import time
import dask.array as da 

class AFNOZarrObsDatasetOptimized(Dataset):
    """
    AFNO PyTorch Dataset:
    - Input from .zarr (CAMx simulation)
    - Target from .npz (observations)
    - On-the-fly PM10 / PM25 computation
    - Optimized for large-scale training
    """

    def __init__(self, sim_zarr_path, obs_npz_path, input_vars, output_vars, species_vars, start_idx=0, end_idx=None):
        super().__init__()

        print("Opening ZARR...")
        self.ds = xr.open_zarr(sim_zarr_path, consolidated=False)
        print("ZARR opened!")

        print("Loading obs NPZ...")
        obs_data = np.load(obs_npz_path, allow_pickle=True)
        print("NPZ loaded!")

        self.input = self.ds["input"]
        #ts_sim = self.ds["timestamps"].values.view("S32").astype(str)
        ts_sim = [s.decode("utf-8").rstrip("\x00") if isinstance(s, bytes) else str(s) for s in self.ds["timestamps"].values]
        #ts_sim = ts_sim.flatten().astype(str).tolist()
        self.obs_target = obs_data["obs_target"]
        ts_obs = obs_data["timestamps"]
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.species_vars = species_vars

        # Match timestamps
        self.pairs = [(i, ts_obs.tolist().index(ts)) for i, ts in enumerate(ts_sim) if ts in ts_obs]

        if not self.pairs:
            raise ValueError("No matching timestamps found between simulation and observation")

        self.start_idx = start_idx
        self.end_idx = end_idx or len(self.pairs)

        self.expanded_vars = self.expand_vars(self.input_vars, self.species_vars)

        self.is_pm10_mode = ("PM10" in input_vars)
        self.is_pm25_mode = ("PM25" in input_vars)


        print(f"AFNO Zarr+Obs Optimized Dataset ready: {len(self.pairs)} matched samples")
        # Timestamps da simulation

        sim_timestamps = set(ts_sim)
        obs_timestamps = set(ts_obs)

        missing_in_obs = sorted(sim_timestamps - obs_timestamps)
        #print(f"Forecast senza osservazioni: {len(missing_in_obs)} samples")
        #print(missing_in_obs[:10])  # primi 10 mancanti

        missing_in_sim = sorted(obs_timestamps - sim_timestamps)
        #print(f"Osservazioni senza forecast: {len(missing_in_sim)} samples")
        #print(missing_in_sim[:10])

    def __len__(self):
        return self.end_idx - self.start_idx

    def compute_pm(self, input_array, species_list):
        pm = np.zeros_like(input_array[0])
        for idx, var in enumerate(self.input_vars):
            if var in species_list:
                pm += input_array[idx]
        return pm
    
    def expand_vars(self, var_list, species_vars):
        expanded = []
        for var in var_list:
            if var == "PM10":
                expanded += species_vars["PM25"] + species_vars["PM10_extra"]
            elif var == "PM25":
                expanded += species_vars["PM25"]
            else:
                expanded.append(var)
        return expanded

    def process_input(self, input_raw):
        if self.is_pm10_mode:
            pm10 = self.compute_pm(input_raw, self.species_vars["PM25"] + self.species_vars["PM10_extra"])
            return np.expand_dims(pm10, axis=0)  # shape [1, H, W]

        elif self.is_pm25_mode:
            pm25 = self.compute_pm(input_raw, self.species_vars["PM25"])
            return np.expand_dims(pm25, axis=0)  # shape [1, H, W]

        else:
            out_vars = []
            for var in self.input_vars:
                try:
                    idx = self.expanded_vars.index(var)
                    out_vars.append(input_raw[idx])
                except ValueError:
                    raise ValueError(f"Variabile {var} non trovata in expanded_vars.")
            return np.stack(out_vars)
    '''
        
    def __getitem__(self, idx):
        sim_idx, obs_idx = self.pairs[idx + self.start_idx]

        arr = self.input[sim_idx].data           # può essere dask.Array
        if isinstance(arr, da.Array):            # ⇢ converto → NumPy
            arr = arr.compute()                  # blocca il chunk in RAM

        T = len(self.input_vars)
        arr = arr.reshape(-1, T, *arr.shape[-2:])  # [TW, T, H, W]
        arr = self.process_input(arr)              # vectorizzato!

        x = torch.as_tensor(arr, dtype=torch.float32) \
                .contiguous(memory_format=torch.channels_last)
        y = torch.as_tensor(self.obs_target[obs_idx], dtype=torch.float32)
        return x, y
    '''
    def __getitem__(self, idx):
        #start = time.time()
        print(f"Fetching idx {idx}")
        sim_idx, obs_idx = self.pairs[idx + self.start_idx]
        
        input_raw = self.input[sim_idx].values  # [T*V, H, W]
        #print("InputRAW:",input_raw.shape)
        T_V, H, W = input_raw.shape
        T = len(self.input_vars)

        input_raw = input_raw.reshape(-1, T, H, W)

        input_processed = []
        for t in range(input_raw.shape[0]):
            input_processed.append(self.process_input(input_raw[t]))

        input_processed = np.concatenate(input_processed, axis=0)

        target = self.obs_target[obs_idx]

        return (
            torch.from_numpy(input_processed).float(),
            torch.from_numpy(target).float()
        )

    #'''