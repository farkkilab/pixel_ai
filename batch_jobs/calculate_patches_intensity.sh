#!/bin/bash
#SBATCH --job-name=create_patches
#SBATCH --account=project_2003009
#SBATCH --time=02:00:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load pytorch/2.1
source venv/bin/activate
srun python scripts/calculate_intensity_allpatches.py --patches_path /scratch/project_2003009/NKI_project_TMAs/patches/randomly_generated/
