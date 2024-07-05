#!/bin/bash
#SBATCH --job-name=gigapath_slides
#SBATCH --account=project_2003009
#SBATCH --time=8:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:2,nvme:500


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.3
source venv/bin/activate

srun python scripts/gigapath_slide_encoder.py --files_path /scratch/project_2003009/NKI_histoprep_tiles --tiles_embedding_path /scratch/project_2003009/histoprep_embeddings --hf_cache_path /scratch/project_2003009/hf_cache --slide_embedding_path /scratch/project_2003009/NKI_slide_embeddings
