#!/bin/bash
#SBATCH --job-name=train_scintilla
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16       
#SBATCH --time=48:00:00          

echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_PER_TASK CPU cores"
echo "Using GPU $CUDA_VISIBLE_DEVICES"

CURR_DIR="/home/gmellone/afno-scintilla"

source "${CURR_DIR}/venv/bin/activate"

# (Opzionale) mostra info GPU
nvidia-smi

# Lancia training
python "${CURR_DIR}/train_afno.py"
