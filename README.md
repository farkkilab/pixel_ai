# Pixel_AI

This repository contains scripts designed for image analysis at a pixel/patch-level resolution using deep learning methods.


### Installation
Install packages listed in requirements.txt. I personally prefer to use a virtualenv to install the packages, but that is up to you.

### Folder structure
* batch_jobs: scripts to run jobs in CSC supercomputers. In most cases, consist of defining hardware resources required in CSC and then calls a script in the `scripts` folder. For example, to run a job inside Mahti, this scripts could be called with "sbatch batch_jobs/run_uniencoder.sh"
* data: miscellaneous data required for models logic. For example, channels list for a dataset.
* models: models architecture definition
* notebooks: Jupyter notebooks for exploring data
* scripts: This includes logic to train model, crop images into tiles, encode tiles, etc. 

### Pipelines
In this repository there are several different pipelines that could be run. To mention some of the main ones:

#### CyCIF pipeline
In this example, using TMAs cores
1) Create tiles, using histoprep (scripts/histoprep_tiles.py)
2) Create an encoding of those tiles (scripts/uni_encoder.py or scripts/gigapath_encoder.py)
3) Train a MIL model using cross validation (scripts/cores_tiles_mil_classifier_cv.py)
4) Adapt logic in notebook for visualizing MIL attention weights (notebooks/mil_model_test.ipynb)

#### SlideFlow pipeline for H&E
NOTE, if slideflow project already exists in the same path, new path should be provided or existing folder should be deleted
1) Create slideflow project and extract tiles (scripts/slide_flow_create_project.py, if in Mahti, batch_jobs/run_slideflow_project.sh)
2) Extract features and train model (scripts/slide_flow_train.py, if in Mahti, batch_jobs/run_slideflow_train.sh)
3) Get attention activations from project folders, exact location is provided by training script, for example: /scratch/project_2003009/he_space_slideflow_cropped/mil/0000 3-attention_mil-er_status_by_ihc/heatmaps