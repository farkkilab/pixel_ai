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
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader, get_percentiles_normalize, PercentileNormalize
import torch
import numpy as np
from multiprocessing import Pool, set_start_method, cpu_count

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    # wh "/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated"
    parser.add_argument("--files_path", type=Path,
                        default="/data/projects/pixel_project/datasets/Launonen_TMA/patches/histoprep_generated/224")
    parser.add_argument("--tiles_embedding_path", type=Path,
                        default="F")
    parser.add_argument("--hf_cache_path", type=Path,
                        default="/data/projects/pixel_project/huggingface/cache")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="mhcii_whole_slide")
    parser.add_argument("--image_normalization", type=str,
                        default=False)
    p = parser.parse_args()
    files_path = p.files_path
    tiles_embedding_path = p.tiles_embedding_path
    data_type = p.data_type
    hf_cache_path = p.hf_cache_path
    image_normalization = p.image_normalization
    if image_normalization == 'true':
        image_normalization = True
    else:
        image_normalization = False
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_path)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    #transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    #transform = transforms.Compose(
    #[
    #    transforms.Resize(224),
    #    transforms.CenterCrop(224),
    #]
#)
    model = model.to(device)
    model.eval()
    #channels = [0, 25, 28]
    #channels = [10, 11, 12]
    channels = [0, 11, 24]
    # Launonen tma
    #channels = [0, 30, 34]
    channels_directory_output = '_'.join(map(str, channels))+'_normalization_'+str(image_normalization)
    tiles_embedding_path = os.path.join(tiles_embedding_path, channels_directory_output)
    batch_size = 128
    num_workers = 28
    labels = []
    # Only files that we have a label for
    files_list = []
    cores = []
    if data_type == 'cores':
        cores_files = []
        slides_directories = [d for d in os.listdir(files_path) if
                              os.path.isdir(os.path.join(files_path, d)) and d.startswith('TMA')]
        for i, slide in enumerate(slides_directories):
            cores_files_path = str(files_path) + "/" + slide
            cores_directories = [d for d in os.listdir(cores_files_path) if
                                 os.path.isdir(os.path.join(cores_files_path, d))]
            for core in cores_directories:
                cores.append(os.path.join(cores_files_path, core))
                cores_files.extend([os.path.join(r, fn)
                                    for r, ds, fs in os.walk(os.path.join(cores_files_path, core))
                                    for fn in fs if fn.endswith('.tiff')])
    elif data_type == 'mhcii_whole_slide':
        cores_files = []
        slides_directories = [d for d in os.listdir(files_path) if
                              os.path.isdir(os.path.join(files_path, d)) and d.startswith('S')]
        for i, slide in enumerate(slides_directories):
            slide_files_path = str(files_path) + "/" + slide +"/tiles"
            for r, ds, fs in os.walk(slide_files_path):
                for fn in fs:
                    cores_files.append(os.path.join(slide_files_path, fn))
    elif data_type == 'whole_slide':
        wholeslide_files = [files_path + slide_dir + '/tiles/' + file for slide_dir in os.listdir(files_path) for file
                            in os.listdir(files_path + '/' + slide_dir + '/tiles') if
                            os.path.isfile(files_path + slide_dir + '/tiles/' + file) and file.endswith('tiff')]
        wholeslide_labels_df = pd.read_csv('data/wholeslide_clinical_data.csv')
    if image_normalization:
        percentile_min, percentile_max, mean, std  = get_percentiles_normalize(cores, channels,
                                                                   min_percentil=1, max_percentil=99)
        print(percentile_min, percentile_max, mean, std)
        normalize_transform = transforms.Compose(
            [
                PercentileNormalize(percentile_min, percentile_max, mean, std, normalization_strategy="min_max"),
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ]
        )
    else:
        normalize_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)
    tiff_dataset = TiffDataset(files=cores_files, files_names=cores_files, transform=normalize_transform, channels=channels)
    sampler = None
    loader = MultiEpochsDataLoader(
        tiff_dataset, batch_size=batch_size, shuffle=(sampler is None),
        pin_memory=True, sampler=sampler, num_workers=num_workers)
    existing_dirs = []
    for i, (images, file_names) in enumerate(loader):
        images = images.cuda()
        with torch.no_grad():
            output = model(images)
        output = output.cpu()
        with Pool(num_workers) as pool:
            args = [(output[i], file_names[i], tiles_embedding_path) for i in range(len(output))
                    if not os.path.exists(os.path.join(tiles_embedding_path,
                                    '/'.join(file_names[i].split('/')[-3:]).replace('.tiff', '') + '_tensor.pt')) ]
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