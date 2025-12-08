#!/bin/bash -l

# SETUP RESOURCE
#SBATCH -A csci4521
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gjers043@umn.edu
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=exports/msi_run.txt

# Locate Conda Profile and Environment
source ~/.bashrc
source /users/6/gjers043/anaconda3/etc/profile.d/conda.sh
conda activate main-env

# Load Modules
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

# Run Script
cd /users/6/gjers043/umn-fall2025-csci5523-project/
CUDA_VISIBLE_DEVICES=0 python3 -m src.main
CUDA_VISIBLE_DEVICES=0 python3 -m src.analysis
