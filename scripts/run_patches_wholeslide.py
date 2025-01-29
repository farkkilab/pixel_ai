import pathlib
import multiprocessing
import tifffile
import os, sys, glob
import ipdb
import zarr
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import create_random_patches, create_random_patches_wholeslide


patch_size = (128, 128)
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=Path,
                    default="/data/projects/sciset/patches/randomly_generated")
parser.add_argument("--slides_path", type=Path,
                    default="/data/projects/sciset/registration/")

p = parser.parse_args()
output_path = p.output_path
slides_path = p.slides_path

# With newer images, we dont need to exclude cores anymore
#cores_ignore_18_810 = open('data/not_good_quality_crops.txt').read().splitlines()
#cores_ignore_18_810 = [core.split('/')[-1].lower() for core in cores_ignore_18_810]

if __name__ == "__main__":
     # List of files to process
    wholeslide_files = [os.path.join(slides_path, d) for d in os.listdir(slides_path) if
                         os.path.isfile(os.path.join(slides_path, d)) and d.endswith('tif')]
    for slide in wholeslide_files:
        pathlib.Path(str(output_path)+'/' + pathlib.Path(slide).stem).mkdir(parents=True, exist_ok=True)
    for slide in wholeslide_files:


        # Number of processes to run in parallel
        num_processes = 128
        image = tifffile.imread(slide)
        image = tifffile.TiffFile(slide, is_ome=False)
        z = zarr.open(image.aszarr(), mode='r')
        num_channels, image_height, image_width = z[0].shape
        # Ensure the patch size is valid
        if patch_size[0] > image_height or patch_size[1] > image_width:
            raise ValueError("Patch size is larger than image dimensions")
        # Create a multiprocessing Pool
        #pool = multiprocessing.Pool(processes=num_processes)
        coordinates = [(top_left_y, top_left_x) for top_left_y in range(0, image_height - patch_size[0] + 1, patch_size[0]) for top_left_x in range(0, image_width - patch_size[1] + 1, patch_size[1])]
        # Use the Pool.starmap function to distribute the work among processes
        [create_random_patches_wholeslide(z, patch_size, str(output_path) + '/' + pathlib.Path(slide).stem, coordinate[0], coordinate[1]) for coordinate
         in coordinates]
        # pool.starmap(create_random_patches_wholeslide, [(z, patch_size, str(output_path)+'/' + pathlib.Path(slide).stem,coordinate[0], coordinate[1]) for coordinate in coordinates])

        # Close the Pool to free up resources
        #pool.close()
        #pool.join()

