#!/bin/bash
#SBATCH --job-name=scintilla_dataset
#SBATCH --output=logs/interpolate.out
#SBATCH --error=logs/interpolate.err
#SBATCH --partition=gpu


CURR_DIR="/home/gmellone/afno-scintilla"

source "${CURR_DIR}/venv/bin/activate"

python "${CURR_DIR}/preprocess/observation_dataset.py"