import pandas as pd
import os, sys, glob
import ipdb
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

patches_output_path = '/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/TMA_18_810/'



if __name__ == "__main__":
    cores_labels_df = pd.read_csv('data/cores_labels.csv')
    patch_labels = {}
    for root, dirs, files in os.walk(patches_output_path):
        for file in files:
            core = root.split('/')[-1]
            slide = root.split('/')[-2]
            patch = os.path.join(root, file)
            cores_label = cores_labels_df[(cores_labels_df['cycif.slide']==slide)&(cores_labels_df['cycif.core.id']==core)]
            if not cores_label.empty:
                patch_labels[patch] = cores_label.iloc[0]['Molecular.profile2']
    patch_labels_df = pd.DataFrame(patch_labels.items(), columns=['patch', 'label'])
    patch_labels_df.to_csv('data/patches_labels.csv', index=False)

