import os

import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from PIL import Image
from torchvision import transforms
import os, sys, glob
import ipdb
from pathlib import Path
import pathlib
import torch.nn as nn
import argparse
import pandas as pd
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader
import torch
from multiprocessing import Pool, set_start_method, cpu_count

def main():
    parser = argparse.ArgumentParser()
    # wh "/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated"
    parser.add_argument("--files_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated")
    parser.add_argument("--tiles_embedding_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_embeddings_uni")
    parser.add_argument("--hf_cache_path", type=Path,
                        default="/data/projects/pixel_project/huggingface/cache")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="cores")
    p = parser.parse_args()
    files_path = p.files_path
    tiles_embedding_path = p.tiles_embedding_path
    data_type = p.data_type
    hf_cache_path = p.hf_cache_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_path)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    #transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)
    model = model.cuda()
    model.eval()
    channels = [0, 25, 28]
    batch_size = 128
    num_workers = 28
    labels = []
    # Only files that we have a label for
    files_list = []

    if data_type == 'cores':
        cores_files = []
        slides_directories = [d for d in os.listdir(files_path) if
                              os.path.isdir(os.path.join(files_path, d)) and d.startswith('TMA')]
        for i, slide in enumerate(slides_directories):
            cores_files_path = str(files_path) + "/" + slide
            cores_directories = [d for d in os.listdir(cores_files_path) if
                                 os.path.isdir(os.path.join(cores_files_path, d))]
            for core in cores_directories:
                cores_files.extend([os.path.join(r, fn)
                                    for r, ds, fs in os.walk(os.path.join(cores_files_path, core))
                                    for fn in fs if fn.endswith('.tiff')])
        cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
        cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')
    elif data_type == 'whole_slide':
        wholeslide_files = [files_path + slide_dir + '/tiles/' + file for slide_dir in os.listdir(files_path) for file
                            in os.listdir(files_path + '/' + slide_dir + '/tiles') if
                            os.path.isfile(files_path + slide_dir + '/tiles/' + file) and file.endswith('tiff')]
        wholeslide_labels_df = pd.read_csv('data/wholeslide_clinical_data.csv')
    if data_type == 'cores':
        for i, core_file in enumerate(cores_files):
            patch_file_label_df = cores_chemo_labels_df[
                (cores_chemo_labels_df['cycif.slide'] == core_file.split('/')[-3]) & (
                            cores_chemo_labels_df['cycif.core.id'] == core_file.split('/')[-2].replace('.tiff', ''))]
            # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
            if not patch_file_label_df.empty and str(
                    patch_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                    patch_file_label_df.iloc[0]['therapy_sequence']):

                files_list.append(core_file)
                # If contains NACT, is a sample collected after chemotherapy exposure
                if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                print('Missing label for:' + core_file)

    tiff_dataset = TiffDataset(files=files_list, files_names=files_list, transform=transform, channels=channels,
                               labels=labels)
    sampler = None
    loader = MultiEpochsDataLoader(
        tiff_dataset, batch_size=batch_size, shuffle=(sampler is None),
        pin_memory=True, sampler=sampler, num_workers=num_workers)
    existing_dirs = []
    for i, (images, file_names, labels) in enumerate(loader):
        images = images.cuda()
        with torch.no_grad():
            output = model(images)
        output = output.cpu()
        with Pool(num_workers) as pool:
            args = [(output[i], file_names[i], tiles_embedding_path) for i in range(len(output))]
            pool.starmap(save_embedding, args)

def save_embedding(embedding, file_name, tiles_embedding_path):
    pathlib.Path(pathlib.Path(os.path.join(tiles_embedding_path, '/'.join(file_name.split('/')[-3:-1])))).mkdir(parents=True,
                                                                                                           exist_ok=True)
    torch.save(embedding, os.path.join(tiles_embedding_path,
                                    '/'.join(file_name.split('/')[-3:]).replace('.tiff', '') + '_tensor.pt'))


if __name__ == "__main__":
    try:
        set_start_method('spawn')
        print("spawned")
    except RuntimeError:
        pass
    main()