import tifffile
import tqdm
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os, sys, glob
import histoprep.functional as F
from histoprep._data import SpotCoordinates, TileCoordinates
import pathlib
import ipdb
from multiprocessing import Pool, set_start_method, cpu_count
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import get_tissue_mask, save_tile


def crop_center(image, crop_height, crop_width):
    """Crops the center of the image to the specified dimensions."""
    height, width = image.shape[1:]
    start_y = max(0,(width - crop_height) // 2)
    start_x = max(0,(height - crop_width) // 2)
    return image[:,start_y:start_y+ min(height,crop_height), start_x:start_x+min(width,crop_width)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path,
                        default="/data/projects/pixel_project/datasets/Launonen_TMA/patches/histoprep_generated")
    parser.add_argument("--slides_path", type=Path,
                        default="/data/data_cloud_mount/farkkila/Data/TMA1/data_cores/TMA1")
    parser.add_argument("--tiles_width", type=int,
                        default=224)
    parser.add_argument("--tiles_height", type=int,
                        default=224)
    p = parser.parse_args()
    output_path = p.output_path
    slides_path = p.slides_path
    tiles_width = p.tiles_width
    tiles_height = p.tiles_height
    output_path = str(output_path)+"/"+str(tiles_width)
    num_workers = 28
    crop_height = 2300
    crop_width = 2300

    #crop_size = 2048
    # channels, DNA1, Vimentin, CK7 (in that order)
    # 0, 25, 28
    # select certain channels to get the tissue mask
    #selected_channels = [0, 25, 28]
    # Only with DAPI
    selected_channels = [0]
    if str(slides_path) == "/data/data_cloud_mount/farkkila/Data/TMA1/data_cores/TMA1":
        slides_directories = ["/data/data_cloud_mount/farkkila/Data/TMA1/data_cores/TMA1"]
    else:
        slides_directories = [d for d in os.listdir(slides_path) if
                                     os.path.isdir(os.path.join(slides_path, d)) and d.startswith('TMA')]
    for slide in slides_directories:
        if str(slides_path) == "/data/data_cloud_mount/farkkila/Data/TMA1/data_cores/TMA1":
            files_to_process = [file for file in glob.glob(str(slides_path) + "/*.tif")]
            output_path_core = str(output_path) + "/TMA1/"
        else:
            files_to_process = [file for file in glob.glob(str(slides_path) + '/' + slide + "/Channels_all/*.tif")]
            output_path_core = str(output_path) + '/' + slide + "/"
        for file_name in files_to_process:
            pathlib.Path(output_path_core + pathlib.Path(file_name).stem).mkdir(parents=True, exist_ok=True)
            im_tiff = tifffile.imread(file_name,level=-1, maxworkers=32)

            # Crop the center of the image
            im_tiff = crop_center(im_tiff, crop_height, crop_width)
            # Determine the dimensions of the image
            #height, width = im_tiff.shape[1:]

            # Calculate the coordinates for the center crop

            #start_x = max(0,(width - crop_size) // 2)
            #start_y = max(0,(height - crop_size) // 2)
            #image_np = im_tiff[selected_channels,start_y:start_y + min(height,crop_size), start_x:start_x + min(width,crop_size)]
            image_np = im_tiff[selected_channels, :,:]
            # Define a threshold to help the tissue detection process
            image_np[image_np < 500] = 0
            image_np = image_np.astype(np.uint8).transpose(1, 2, 0)
            # Convert the image to 3 channels
            image_np = np.dstack((image_np, image_np, image_np))



            output_dir = output_path_core + pathlib.Path(file_name).stem
            threshold, tissue_mask = get_tissue_mask(image_np)
            print(slide, file_name)
            # Extract overlapping tile coordinates with less than 50% background.
            overlap = 0.1
            tile_coordinates = F.get_tile_coordinates(
                tissue_mask.shape, height=tiles_height,width=tiles_width, overlap=overlap
            )
            max_background = 0.75
            if tissue_mask is not None:
                all_backgrounds = F.get_background_percentages(
                    tile_coordinates=tile_coordinates,
                    tissue_mask=tissue_mask,
                    downsample=1,
                )
                filtered_coordinates = []
                for xywh, background in zip(tile_coordinates, all_backgrounds):
                    if background <= max_background:
                        filtered_coordinates.append(xywh)
                tile_coordinates = filtered_coordinates
                tiles = TileCoordinates(
                coordinates=tile_coordinates,
                width=tiles_width,
                height=tiles_width if tiles_height is None else tiles_height,
                overlap=overlap,
                max_background=max_background,
                tissue_mask=tissue_mask,
                )
                kwargs = {
                    "image": image_np,
                    "downsample": 1,
                    "rectangle_width": 1,
                    "rectangle_fill":"red"
                }
                kwargs.update(
                    {"coordinates": tiles.coordinates, "highlight_first": True}
                )
                Image.fromarray(image_np).save(output_dir+ "thumbnail.tiff")
                thumbnail_regions = F.get_annotated_image(**kwargs)
                thumbnail_regions.save(output_dir + f"thumbnail_tiles.tiff")
                Image.fromarray(255 - 255 * tiles.tissue_mask).save(
                    output_dir + "thumbnail_tissue.tiff")
                with Pool(num_workers) as pool:
                    args = [(tile_coordinates[i], im_tiff, output_dir) for i in range(len(tile_coordinates))]
                    pool.starmap(save_tile, args)


if __name__ == "__main__":
    try:
        set_start_method('spawn')
        print("spawned")
    except RuntimeError:
        pass
    main()