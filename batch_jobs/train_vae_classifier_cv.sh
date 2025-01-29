#!/bin/bash
#SBATCH --job-name=vaeclassifier_training
#SBATCH --account=project_2003009
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.3
source venv/bin/activate
python scripts/train_vae_classifier_cv.py
