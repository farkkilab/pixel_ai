import glob, os
import tifffile
import pandas as pd


channels = [0]
images_values = []
path = '/data/projects/sciset/patches/randomly_generated'

for root, directories, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
                im_tiff = tifffile.imread(os.path.join(root, file), key=channels, maxworkers=32)
                images_values.append({'file_name': os.path.join(root, file), 'total_value': im_tiff.sum()})
images_values_df = pd.DataFrame(images_values)
images_values_df.to_csv('data/wholeslide_patches_statistics.csv', index=False)
