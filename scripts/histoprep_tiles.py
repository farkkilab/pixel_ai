import tifffile
import tqdm
import numpy as np
from PIL import Image
import os, sys, glob
import histoprep.functional as F
from histoprep._data import SpotCoordinates, TileCoordinates
import ipdb
from multiprocessing import Pool, set_start_method, cpu_count
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import get_tissue_mask, save_tile



def main():
    # channels, DNA1, Vimentin, CK7 (in that order)
    # 0, 38, 42
    # select certain channels to get the tissue mask
    selected_channels = [0]
    slides_path = '/data/projects/sciset/registration/'
    slides_to_process = [file for file in glob.glob(str(slides_path)+"*.tif")]
    for slide in slides_to_process:
        im_tiff = tifffile.imread(slide,level=-1, maxworkers=32)
        output_dir = "/data/projects/sciset/train_tiles/"+slide.split('/')[-1].split('.')[0]+'/'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir+'tiles', exist_ok=True)
        image_np = im_tiff[selected_channels,:,:].astype(np.uint8).transpose(1, 2, 0)
        # Convert the image to 3 channels
        image_np = np.dstack((image_np, image_np, image_np))

        threshold, tissue_mask = get_tissue_mask(image_np)
        # Extract overlapping tile coordinates with less than 50% background.
        width = 224
        height = 224
        num_workers = 28
        overlap = 0.1
        tile_coordinates = F.get_tile_coordinates(
            tissue_mask.shape, height=height,width=width, overlap=0.5
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
            width=width,
            height=width if height is None else height,
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