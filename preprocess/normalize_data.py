import os
import yaml
import numpy as np

sim_stats_path = "/storage/external_01/scintilla/processed_afno/training_2017/normalization_stats.yaml"
obs_stats_path = "/storage/external_01/scintilla/processed_afno/training_2017/obs/normalization_stats_obs.yaml"
out_path = "/storage/external_01/scintilla/processed_afno/training_2017/normalization_stats_combined.yaml"

# Numero di esempi usati per calcolare ciascun file (personalizzabili)
n_sim = 10000  # oppure caricalo dinamicamente se disponibile
n_obs = 10000

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def combine_mean_std(mean1, std1, n1, mean2, std2, n2):
    mean_comb = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    var_comb = (n1 * std1**2 + n2 * std2**2) / (n1 + n2)
    std_comb = np.sqrt(var_comb)
    return float(mean_comb), float(std_comb)

def combine_stats(sim_stats, obs_stats, n_sim, n_obs):
    combined = {"input": {}, "target": {}}
    for section in ["input", "target"]:
        sim_section = sim_stats.get(section, {})
        obs_section = obs_stats.get(section, {})
        keys = set(sim_section.keys()).union(obs_section.keys())

        for key in keys:
            s = sim_section.get(key)
            o = obs_section.get(key)
            if s and o:
                mean, std = combine_mean_std(s["mean"], s["std"], n_sim, o["mean"], o["std"], n_obs)
            elif s:
                mean, std = s["mean"], s["std"]
            elif o:
                mean, std = o["mean"], o["std"]
            else:
                continue
            combined[section][key] = {"mean": mean, "std": std}
    return combined

def main():
    sim_stats = load_yaml(sim_stats_path)
    obs_stats = load_yaml(obs_stats_path)
    combined_stats = combine_stats(sim_stats, obs_stats, n_sim, n_obs)

    with open(out_path, "w") as f:
        yaml.dump(combined_stats, f)
    print(f"✅ Combined normalization stats saved to: {out_path}")

if __name__ == "__main__":
    main()
