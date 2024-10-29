#!/bin/bash
#SBATCH --job-name=wsi_mil_training
#SBATCH --account=project_2003009
#SBATCH --time=36:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1,nvme:900

rsync -av --progress /scratch/project_2003009/MHCII_histoprep_embeddings_uni/224 $LOCAL_SCRATCH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load pytorch/2.3

source venv/bin/activate
srun python scripts/wsi_tiles_mil_classifier.py --files_path $LOCAL_SCRATCH/0_0_0_normalization_False --model_encoder uni --image_normalization false
