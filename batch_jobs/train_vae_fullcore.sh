#!/bin/bash
#SBATCH --job-name=vaefullcore_training
#SBATCH --account=project_2003009
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:2,nvme:600

rsync -av --progress /scratch/project_2003009/NKI_project_TMAs $LOCAL_SCRATCH --exclude patches


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.1
source venv/bin/activate
python scripts/train_vanilla_vae_fullcore.py --cores_path $LOCAL_SCRATCH/NKI_project_TMAs --latent_dims 16 --lr 0.00001
python scripts/train_vanilla_vae_fullcore.py --cores_path $LOCAL_SCRATCH/NKI_project_TMAs --latent_dims 16 --lr 0.000001
python scripts/train_vanilla_vae_fullcore.py --cores_path $LOCAL_SCRATCH/NKI_project_TMAs --latent_dims 16 --lr 0.0000001
