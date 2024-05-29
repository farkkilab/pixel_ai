from histoprep import SlideReader
import ipdb

# Read slide image.
reader = SlideReader("/data/projects/sciset/registration/Sample_01.ome.tif")
# Detect tissue.
threshold, tissue_mask = reader.get_tissue_mask(level=-1)
# Extract overlapping tile coordinates with less than 50% background.
tile_coordinates = reader.get_tile_coordinates(
    tissue_mask, width=512, overlap=0.5, max_background=0.5
)
ipdb.set_trace()
# Save tile images with image metrics for preprocessing.
tile_metadata = reader.save_regions(
    "./train_tiles/", tile_coordinates, threshold=threshold, save_metrics=True
)
