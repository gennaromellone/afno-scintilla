#!/bin/bash
#SBATCH --job-name=scintilla_dataset
#SBATCH --output=logs/build_dataset.out
#SBATCH --error=logs/build_dataset.err
#SBATCH --partition=gpu


CURR_DIR="/home/gmellone/afno-scintilla"

source "${CURR_DIR}/venv/bin/activate"

python "${CURR_DIR}/preprocess/build_camx_training_dataset_parallel.py"