#!/bin/bash
#SBATCH --job-name=vaefullcore_training
#SBATCH --account=project_2003009
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1

#rsync -av --progress /scratch/project_2003009/NKI_project_TMAs $LOCAL_SCRATCH --exclude patches


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.3
source venv/bin/activate
srun python scripts/train_vanilla_vae.py --cores_path /scratch/project_2003009/NKI_project_TMAs
#python scripts/train_vanilla_vae.py --cores_path /scratch/project_2003009/NKI_project_TMAs --latent_dims 16 --lr 0.000001
#python scripts/train_vanilla_vae.py --cores_path /scratch/project_2003009/NKI_project_TMAs --latent_dims 16 --lr 0.0000001
