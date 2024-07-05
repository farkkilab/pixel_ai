#!/bin/bash
#SBATCH --job-name=histoprep_tiles
#SBATCH --account=project_2003009
#SBATCH --time=12:00:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#cp -r /scratch/project_2003009/NKI_project_TMAs $LOCAL_SCRATCH/NKI_project_TMAs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load pytorch/2.3

source venv/bin/activate
srun python scripts/histoprep_cores_tiles.py --slides_path /scratch/project_2003009/NKI_project_TMAs --output_path /scratch/project_2003009/NKI_histoprep_tiles
