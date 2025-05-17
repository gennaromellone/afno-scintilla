#!/bin/bash
#SBATCH --job-name=scintilla_dataset
#SBATCH --output=logs/camx_dataset.out
#SBATCH --error=logs/camx_dataset.err
#SBATCH --partition=gpu


CURR_DIR="/home/gmellone/afno-scintilla"

source "${CURR_DIR}/venv/bin/activate"

python "${CURR_DIR}/preprocess/camx_dataset.py"