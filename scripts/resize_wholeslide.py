import pyvips
from pathlib import Path
import argparse
from skimage.transform import resize
import cv2
from joblib import Parallel, delayed
import ipdb
import tifffile
import numpy as np
import os, sys, glob


parser = argparse.ArgumentParser()
parser.add_argument("--files_path", type=Path,
                    default="/data/projects/sciset/registration/")
parser.add_argument("--output_path", type=Path,
                    default="/data/projects/sciset/resized/")
p = parser.parse_args()
files_path = p.files_path
output_path = p.output_path

wholeslide_files = [os.path.join(files_path, d) for d in os.listdir(files_path) if
                             os.path.isfile(os.path.join(files_path, d)) and d.endswith('tif')]


def resize_image(file):
    channels = [0, 12, 28]
    im_tiff = tifffile.imread(file,key=channels, maxworkers=32)
    im_tiff = np.moveaxis(im_tiff, 0, -1)
    aspect_ratio = im_tiff.shape[1] / im_tiff.shape[0]

    # Calculate the new width based on the maximum height of 5000 pixels
    new_height = min(im_tiff.shape[0], 5000)
    new_width = int(new_height * aspect_ratio)
    resized_data = cv2.resize(im_tiff, (new_width, new_height))  # cv2.resize(image,(width,height))
    resized_data = np.moveaxis(resized_data, -1, 0)
    #resized_data = np.moveaxis(resized_data, 1, -1)
    #resized_data = resize(im_tiff, (im_tiff_shape[1], im_tiff_shape[2], im_tiff_shape[0]))
    tifffile.imwrite(str(output_path) + '/' + file.split('/')[-1], resized_data)

def joblib_loop():
    Parallel(n_jobs=1)(delayed(resize_image)(i) for i in wholeslide_files)

if __name__ == '__main__':
    for file in wholeslide_files:
        resize_image(file)








