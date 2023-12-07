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
import os, sys
import torchvision.transforms as T
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from utils import create_random_patches, ProgressMeter, AverageMeter, save_checkpoint, TiffDataset
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname("__file__"), '..')))

PIL.Image.MAX_IMAGE_PIXELS = 933120000

cores_folder = 'TMA_18_810/'
files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
input_dimensions = (32, 32)
batch_size = 4096

if __name__ == "__main__":
    patches_files = [os.path.join(r, fn)
            for r, ds, fs in os.walk(files_path)
            for fn in fs if fn.endswith('.tiff')]
    tiff_dataset = TiffDataset(files=patches_files, files_names=patches_files, transform=T.Resize(input_dimensions))

    data_loader = torch.utils.data.DataLoader(
        tiff_dataset, batch_size=batch_size, 
        shuffle=None, pin_memory=True, 
        sampler=None
        )

    dapi_stat = []

    for i, (image, filenames) in enumerate(data_loader): 
        for name in filenames: 
            im_core_patch = tifffile.imread(name)
            dapi_stat.append((np.median(im_core_patch[0,:,:].flatten()), 
                            np.mean(im_core_patch[0,:,:].flatten()), 
                            np.std(im_core_patch[0,:,:].flatten()), 
                            name.split('/')[-1].lower(), 
                            name.split('/')[-2].lower()))

    d = pd.DataFrame(dapi_stat, columns=("Median", "Mean", "Std", "Patch", "Core"))

    # save the file
    d.to_csv("../data/patch_stat_channel0.csv")