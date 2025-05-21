import os
import yaml
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_normalization_stats(path):
    """Carica statistiche di normalizzazione da file YAML"""
    if not os.path.exists(path):
        logger.error(f"Normalization file not found: {path}")
        return None

    with open(path) as f:
        stats = yaml.safe_load(f)
    logger.info(f"📦 Loaded normalization stats from: {path}")
    return stats


def apply_normalization(array, stats, names, kind="input"):
    """
    Normalizza un array 3D (C, H, W) o 4D (T, C, H, W) usando le statistiche fornite.

    :param array: np.array da normalizzare
    :param stats: dizionario delle statistiche
    :param names: lista di nomi canali corrispondenti (es. ["PM10", "NO2"] o ["ch_0", "ch_1"])
    :param kind: "input" o "target", usato per log
    :return: array normalizzato
    """
    for i, name in enumerate(names):
        if name not in stats:
            logger.warning(f"⚠️  No stats found for {kind} channel '{name}' – skipping normalization")
            continue

        mean = stats[name].get("mean", 0.0)
        std = stats[name].get("std", 1.0)
        if std > 0:
            array[i] = (array[i] - mean) / std
        else:
            logger.warning(f"⚠️  Std = 0 for {kind} channel '{name}' – skipping normalization")
    return array
