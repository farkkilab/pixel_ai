import pathlib
import multiprocessing
import os, sys, glob
import ipdb
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import create_random_patches


patch_size = (128, 128)
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=Path,
                    default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/")
parser.add_argument("--slides_path", type=Path,
                    default="/data/projects/pixel_project/datasets/NKI_project_TMAs/")

p = parser.parse_args()
output_path = p.output_path
slides_path = p.slides_path

# With newer images, we dont need to exclude cores anymore
#cores_ignore_18_810 = open('data/not_good_quality_crops.txt').read().splitlines()
#cores_ignore_18_810 = [core.split('/')[-1].lower() for core in cores_ignore_18_810]

if __name__ == "__main__":
     # List of files to process
    slides_directories = [d for d in os.listdir(slides_path) if
                             os.path.isdir(os.path.join(slides_path, d)) and d.startswith('TMA')]
    for slide in slides_directories:
        files_to_process = [file for file in glob.glob(str(slides_path)+slide+"/Channels_all/*.tif")]
        ipdb.set_trace()
        output_path_core = str(output_path)+slide+"/"
        for file_name in files_to_process:
            pathlib.Path(output_path_core+pathlib.Path(file_name).stem).mkdir(parents=True, exist_ok=True)

        # Number of processes to run in parallel
        num_processes = 16

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=num_processes)

        # Use the Pool.starmap function to distribute the work among processes
        pool.starmap(create_random_patches, [(file, patch_size, output_path_core + pathlib.Path(file).stem) for file in files_to_process])

        # Close the Pool to free up resources
        pool.close()
        pool.join()

