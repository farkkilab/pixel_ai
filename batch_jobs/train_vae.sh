#!/bin/bash
#SBATCH --job-name=vae_training
#SBATCH --account=project_2003009
#SBATCH --time=1:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1,nvme:500

cp -r /scratch/project_2003009/NKI_project_TMAs/patches $LOCAL_SCRATCH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.1
source venv/bin/activate
srun python -m cProfile -o vanilla_vae_profile scripts/train_vanilla_vae.py --patches_path $LOCAL_SCRATCH/patches/randomly_generated/
