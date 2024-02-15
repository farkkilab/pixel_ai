#!/bin/bash
#SBATCH --job-name=vae_training
#SBATCH --account=project_2003009
#SBATCH --time=24:00:00
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.1
source venv/bin/activate
srun python scripts/train_vanilla_vae.py --patches_path /scratch/project_2003009/NKI_project_TMAs/patches/randomly_generated/
