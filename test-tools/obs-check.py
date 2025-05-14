import numpy as np

data = np.load("data/processed/afno_obs_interpolated.npz", allow_pickle=True)

print("OBS NPZ")
print("obs_target shape:", data["obs_target"].shape)

print("Mean:", np.nanmean(data["obs_target"]))
print("Min:", np.nanmin(data["obs_target"]))
print("Max:", np.nanmax(data["obs_target"]))

print("NaN count:", np.isnan(data["obs_target"]).sum())

print("Sample timestamp:", data["timestamps"][0])
