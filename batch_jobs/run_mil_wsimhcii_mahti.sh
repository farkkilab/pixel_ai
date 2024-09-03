#!/bin/bash
#SBATCH --job-name=wsi_mil_training
#SBATCH --account=project_2003009
#SBATCH --time=8:00:00
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load pytorch/2.3

source venv/bin/activate
srun python scripts/wsi_tiles_mil_classifier.py --files_path /scratch/project_2003009/NKI_wsi_MHCII_histoprep_patches/ --model_encoder trainable --image_normalization false
