#!/bin/bash
#SBATCH --job-name=create_patches
#SBATCH --account=project_2003009
#SBATCH --time=02:00:00
#SBATCH --partition=medium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128

source venv/bin/activate
srun python3 scripts/run_random_patches_creation.py --output_path /scratch/project_2003009/NKI_project_TMAs/patches/randomly_generated/ --slides_path /scratch/project_2003009/NKI_project_TMAs/
