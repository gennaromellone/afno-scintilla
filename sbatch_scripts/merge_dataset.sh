#!/bin/bash
#SBATCH --job-name=scintilla_merge
#SBATCH --output=logs/merge.out
#SBATCH --error=logs/merge.err
#SBATCH --partition=gpu


CURR_DIR="/home/gmellone/afno-scintilla"

source "${CURR_DIR}/venv/bin/activate"

python "${CURR_DIR}/preprocess/merge_chunks_parallel.py"