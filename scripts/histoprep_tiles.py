import tifffile
import tqdm
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os, sys, glob
import histoprep.functional as F
from histoprep._data import SpotCoordinates, TileCoordinates
import ipdb
from multiprocessing import Pool, set_start_method, cpu_count
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import get_tissue_mask, save_tile



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path,
                        default="/data/projects/sciset/train_tiles/")
    parser.add_argument("--slides_path", type=Path,
                        default="/data/projects/sciset/registration/")
    parser.add_argument("--tiles_width", type=int,
                        default=224)
    parser.add_argument("--tiles_height", type=int,
                        default=224)
    p = parser.parse_args()
    output_path = p.output_path
    slides_path = p.slides_path
    tiles_width = p.tiles_width
    tiles_height = p.tiles_height
    # channels, DNA1, Vimentin, CK7 (in that order)
    # 0, 38, 42
    # select certain channels to get the tissue mask
    selected_channels = [0]
    slides_to_process = [file for file in glob.glob(str(slides_path)+"/*.tiff")]
    for slide in slides_to_process:
        with tifffile.TiffFile(slide) as tif:
            page_0 = tif.series[0].levels[0]
            channels_page_0, height_page_0, width_page_0 = page_0.shape
            page_3 = tif.series[0].levels[3]
            channels_page_3, height_page_3, width_page_3 = page_3.shape
            width_ratio = width_page_3 / width_page_0
            height_ratio = height_page_3 / height_page_0
            print(width_ratio, height_ratio)
        im_tiff = page_3.asarray()#tifffile.imread(slide,level=0, maxworkers=32)
        output_dir = str(output_path)+'/'+slide.split('/')[-1].split('.')[0]+'/'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir+'tiles', exist_ok=True)
        image_np = im_tiff[selected_channels,:,:].astype(np.uint8).transpose(1, 2, 0)
        # Convert the image to 3 channels
        image_np = np.dstack((image_np, image_np, image_np))

        threshold, tissue_mask = get_tissue_mask(image_np)
        # Extract overlapping tile coordinates with less than 50% background.
        num_workers = cpu_count() - 1
        overlap = 0.1
        tile_coordinates = F.get_tile_coordinates(
            tissue_mask.shape, height=int(tiles_height*height_ratio),width=int(tiles_width*width_ratio), overlap=0.5
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
            width=int(tiles_width*width_ratio),
            height=int(tiles_width*width_ratio) if tiles_height is None else int(tiles_height*height_ratio),
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
            with Pool(cpu_count()//4) as pool:
                args = [(tile_coordinates[i], tifffile.imread(slide, level=0, maxworkers=32, selection=(slice(0,channels_page_0),slice(int(tile_coordinates[i][1]/width_ratio), int(tile_coordinates[i][1]/width_ratio)+int(tile_coordinates[i][3]/width_ratio)+1),slice(int(tile_coordinates[i][0]/width_ratio), int(tile_coordinates[i][0]/width_ratio)+int(tile_coordinates[i][2]/width_ratio)+1))), output_dir+'/tiles', width_ratio) for i in range(len(tile_coordinates))]
                pool.starmap(save_tile, args)


if __name__ == "__main__":
    try:
        set_start_method('spawn')
        print("spawned")
    except RuntimeError:
        pass
    main()