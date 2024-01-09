import pathlib
import multiprocessing
import os, sys, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import create_random_patches


patch_size = (128, 128)
output_path = '/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/TMA_18_810/'
# With newer images, we dont need to exclude cores anymore
#cores_ignore_18_810 = open('data/not_good_quality_crops.txt').read().splitlines()
#cores_ignore_18_810 = [core.split('/')[-1].lower() for core in cores_ignore_18_810]

if __name__ == "__main__":
     # List of files to process
    files_to_process = [file for file in glob.glob("/data/projects/pixel_project/datasets/NKI_project_TMAs/TMA_18_810/Channels_all/*.tif")]
    for file_name in files_to_process:
        pathlib.Path(output_path+file_name.split('.')[0]).mkdir(parents=True, exist_ok=True)

    # Number of processes to run in parallel
    num_processes = 16

    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the Pool.starmap function to distribute the work among processes
    pool.starmap(create_random_patches, [(file, patch_size, output_path + pathlib.Path(file).stem) for file in files_to_process])

    # Close the Pool to free up resources
    pool.close()
    pool.join()

