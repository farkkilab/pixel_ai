#!/bin/bash
#SBATCH --job-name=gigapath_tiles
#SBATCH --account=project_2003009
#SBATCH --time=36:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1

export HUGGING_FACE_HUB_TOKEN=***REMOVED***
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.3
source venv/bin/activate

#srun python scripts/gigapath_encoder.py --files_path /scratch/project_2003009/NKI_histoprep_patches/224 --hf_cache_path /scratch/project_2003009/hf_cache --tiles_embedding_path /scratch/project_2003009/gigapath_tiles_embeddings/224
#srun python scripts/gigapath_encoder.py --files_path /scratch/project_2003009/Launonen_TMA_patches/224 --hf_cache_path /scratch/project_2003009/hf_cache --tiles_embedding_path /scratch/project_2003009/Launonen_histoprep_embeddings_gigapath/224
srun python scripts/gigapath_encoder.py --files_path /scratch/project_2003009/NKI_wsi_MHCII_histoprep_patches --hf_cache_path /scratch/project_2003009/hf_cache --tiles_embedding_path /scratch/project_2003009/MHCII_histoprep_embeddings_gigapath

