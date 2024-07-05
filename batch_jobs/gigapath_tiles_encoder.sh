#!/bin/bash
#SBATCH --job-name=gigapath_tiles
#SBATCH --account=project_2003009
#SBATCH --time=12:00:00
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4,nvme:500

cp -r /scratch/project_2003009/NKI_histoprep_tiles $LOCAL_SCRATCH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.3
source venv/bin/activate

srun python scripts/gigapath_encoder.py --files_path $LOCAL_SCRATCH/NKI_histoprep_tiles --hf_cache_path /scratch/project_2003009/hf_cache --tiles_embedding_path /scratch/project_2003009/histoprep_embeddings
