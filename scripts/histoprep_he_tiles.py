import tifffile
import tqdm
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os, sys, glob
import openslide
import histoprep.functional as F
from histoprep._data import SpotCoordinates, TileCoordinates
from histoprep import SlideReader
import ipdb
from multiprocessing import Pool, set_start_method, cpu_count
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import get_tissue_mask, save_tile_he



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path,
                        default="/scratch/project_2003009/PKG-PTR-HGSOC_tiles/")
    parser.add_argument("--slides_path", type=Path,
                        default="/scratch/project_2003009/PKG-PTR-HGSOC/data/")
    parser.add_argument("--tiles_width", type=int,
                        default=224)
    parser.add_argument("--tiles_height", type=int,
                        default=224)
    p = parser.parse_args()
    output_path = p.output_path
    slides_path = p.slides_path
    tiles_width = p.tiles_width
    tiles_height = p.tiles_height
    slides_to_process = [file for file in glob.glob(str(slides_path)+"/*.svs*")]
    slides_to_avoid = ['/scratch/project_2003009/PKG-PTR-HGSOC/data/1009786_271252_ImageActual.svs',
                       '/scratch/project_2003009/PKG-PTR-HGSOC/data/1009676_269185_ImageActual.svs']
    for slide in slides_to_process:
        if slide in slides_to_avoid:
            continue
        print(slide)
        slide_data = openslide.OpenSlide(slide)
        reader = SlideReader(slide)
        width, height = slide_data.dimensions
        output_dir = str(output_path)+'/'+slide.split('/')[-1].split('.')[0]+'/'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir+'tiles', exist_ok=True)
        image_np = slide_data.read_region((0, 0), 0, (width, height))
        image_np = image_np.convert("RGB")
        image_np = np.array(image_np)
        #slide_data.close()
        # Define a threshold to help the tissue detection process
        #image_np[image_np < 500] = 0
        #image_np = image_np.astype(np.uint8).transpose(1, 2, 0)
        # Convert the image to 3 channels
        #image_np = np.dstack((image_np, image_np, image_np))

        #threshold, tissue_mask = get_tissue_mask(image_np)
        try:
            threshold, tissue_mask = reader.get_tissue_mask(level=0)
        except:
            continue
        # Extract overlapping tile coordinates with less than 50% background.
        num_workers = cpu_count() - 1
        overlap = 0.1
        tile_coordinates = reader.get_tile_coordinates(
            tissue_mask, height=tiles_height,width=tiles_width, overlap=0.5
        )
        max_background = 0.7
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
            height=tiles_height,
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
            #Image.fromarray(image_np).save(output_dir+ "thumbnail.png")
            try:
                thumbnail_regions = F.get_annotated_image(**kwargs)
                thumbnail_regions.save(output_dir + f"thumbnail_tiles.jpg")
                Image.fromarray(255 - 255 * tiles.tissue_mask).save(
                    output_dir + "thumbnail_tissue.png")
            except:
                continue
            with Pool(16) as pool:
                args = [(tile_coordinates[i], slide, output_dir+'/tiles') for i in range(len(tile_coordinates))]
                pool.starmap(save_tile_he, args)


if __name__ == "__main__":
    try:
        set_start_method('spawn')
        print("spawned")
    except RuntimeError:
        pass
    main()