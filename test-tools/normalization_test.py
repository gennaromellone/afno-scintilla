import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--norm_file", type=str, required=True, help="Path to normalization_stats.yaml")
args = parser.parse_args()

with open(args.norm_file) as f:
    stats = yaml.safe_load(f)

print("\n📊 Normalization Summary:\n")

for section in stats:
    print(f"[{section.upper()}]")
    for var, values in stats[section].items():
        mean = values.get("mean")
        std = values.get("std")
        if std == 0:
            print(f"⚠️  {var}: std=0 → INVALID")
            continue

        real_min = mean - 3 * std
        real_max = mean + 3 * std
        print(f"  • {var:10s} | mean={mean:.3f} | std={std:.3f} | real range ≈ [{real_min:.1f}, {real_max:.1f}] → normal range [-3, +3]")
    print()
