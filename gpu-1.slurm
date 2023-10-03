#! /bin/bash
# ====================================
#SBATCH --job-name=GPU_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
# ====================================

# Check the GPUs with the nvidia-smi command.
nvidia-smi
