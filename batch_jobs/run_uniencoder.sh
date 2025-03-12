#!/bin/bash
#SBATCH --job-name=uniencoder
#SBATCH --account=project_2003009
#SBATCH --time=36:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load pytorch/2.3

source venv/bin/activate
srun python scripts/uni_encoder.py --files_path /scratch/project_2003009/NKI_histoprep_patches/224 --tiles_embedding_path /scratch/project_2003009/histoprep_embeddings_uni/224 --image_normalization false --data_type cores
#srun python scripts/uni_encoder.py --files_path /scratch/project_2003009/Launonen_TMA_patches/224 --tiles_embedding_path /scratch/project_2003009/Launonen_histoprep_embeddings_uni/224 --image_normalization false
#srun python scripts/uni_encoder.py --files_path //scratch/project_2003009/NKI_wsi_MHCII_histoprep_patches/ --tiles_embedding_path /scratch/project_2003009/MHCII_histoprep_embeddings_uni/224 --image_normalization false


#python scripts/uni_encoder.py --files_path /data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated/224 --tiles_embedding_path /data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_embeddings_uni/224 --image_normalization true
