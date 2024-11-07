#!/bin/bash
#SBATCH --job-name=mil_training_cv
#SBATCH --account=project_2003009
#SBATCH --time=4:00:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load pytorch/2.3

source venv/bin/activate
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/gigapath_tiles_embeddings/224/0_25_28 --filter_cores_lowcancer false --model_encoder gigapath --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/gigapath_tiles_embeddings/224/0_25_28 --filter_cores_lowcancer true --model_encoder gigapath --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/gigapath_tiles_embeddings/224/0_0_0 --filter_cores_lowcancer false --model_encoder gigapath --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/gigapath_tiles_embeddings/224/0_0_0 --filter_cores_lowcancer true --model_encoder gigapath --image_normalization false

#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28_normalization_True --filter_cores_lowcancer false --model_encoder uni --image_normalization true
srun python scripts/cores_tiles_mil_classifier_cv.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28 --filter_cores_lowcancer false --model_encoder uni --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_0_0_normalization_False --filter_cores_lowcancer false --model_encoder uni --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28 --filter_cores_lowcancer false --model_encoder trainable --image_normalization true
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28 --filter_cores_lowcancer false --model_encoder trainable --image_normalization false

#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28_normalization_True --filter_cores_lowcancer true --model_encoder uni --image_normalization true
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28_normalization_False --filter_cores_lowcancer true --model_encoder uni --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_0_0_normalization_False --filter_cores_lowcancer true --model_encoder uni --image_normalization false
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28 --filter_cores_lowcancer true --model_encoder trainable --image_normalization true
#srun python scripts/cores_tiles_mil_classifier.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_25_28 --filter_cores_lowcancer true --model_encoder trainable --image_normalization false
#srun python scripts/cores_tiles_mil_classifier_cv.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_0_0_normalization_False --filter_cores_lowcancer false --model_encoder trainable --image_normalization false
#srun python scripts/cores_tiles_mil_classifier_cv.py --files_path /scratch/project_2003009/histoprep_embeddings_uni/224/0_0_0_normalization_False --filter_cores_lowcancer true --model_encoder trainable --image_normalization false
