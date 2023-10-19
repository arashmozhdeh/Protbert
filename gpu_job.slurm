#!/bin/bash
#! /bin/bash
# ====================================
#SBATCH --job-name=STEP-10Shot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
# ====================================

python src/ppi_finetuning.py --accelerator gpu --devices 1 --num_sanity_val_steps 0 --train_csv "/home/arash.mozhdehi/GPU_STEP/STEP-main/data/generated/NewData/train_pairs_10shots.csv" --valid_csv "/home/arash.mozhdehi/GPU_STEP/STEP-main/data/generated/NewData/val_pairs_10shots.csv" --test_csv "/home/arash.mozhdehi/GPU_STEP/STEP-main/data/generated/NewData/test_pairs_10shots.csv" --checkpoint_filename "vp1_ppi_model10.ckpt"
