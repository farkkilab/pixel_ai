from PIL import Image
import PIL
import os
import tifffile
import glob
import torch; torch.manual_seed(0)
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import multiprocessing
import ipdb
import os, sys
import torchvision.transforms as T
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import get_patch_stats, ProgressMeter, AverageMeter, save_checkpoint, TiffDataset


parser = argparse.ArgumentParser()
parser.add_argument("--patches_path", type=Path,
                    default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/")
p = parser.parse_args()
patches_path = p.patches_path
if __name__ == "__main__":
    patches_files = []
    patches_directories = [d for d in os.listdir(patches_path) if
                           os.path.isdir(os.path.join(patches_path, d)) and d.startswith('TMA')]
    for slide in patches_directories:
        files_path = patches_path+slide
        patches_files.extend([os.path.join(r, fn)
                              for r, ds, fs in os.walk(files_path)
                              for fn in fs if fn.endswith('.tiff')])



    for file in patches_files:
        get_patch_stats(file)


    num_processes = 128

    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the Pool.starmap function to distribute the work among processes
    patches_results = pool.starmap(get_patch_stats,
                 [(file,) for file in patches_files])

    # Close the Pool to free up resources
    pool.close()
    pool.join()


    patches_stats_df = pd.DataFrame(patches_results, columns=("Median", "Mean", "Std", "Patch", "Core", "Slide"))

    # save the file
    patches_stats_df.to_csv("data/patch_size_128_stat_channel0.csv")
