#!/bin/bash
#SBATCH --job-name=slideflow_spacehe
#SBATCH --account=project_2003009
#SBATCH --time=4:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1


#cp -r /scratch/project_2003009/NKI_project_TMAs $LOCAL_SCRATCH/NKI_project_TMAs
export TRANSFORMERS_CACHE=/scratch/project_2003009/transformers_cache
export HF_HOME=/scratch/project_2003009/transformers_cache
module load pytorch/2.3

export SF_SLIDE_BACKEND=libvips
export SING_FLAGS=--nv
export APPTAINERENV_LD_LIBRARY_PATH=/usr/local/lib/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib:/.singularity.d/libs

source venv/bin/activate

srun apptainer exec -B /scratch:/scratch slideflow.sif python scripts/slide_flow_train.py

