import os

#os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/projects/pixel_project/huggingface/cache"
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
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated/224")
    parser.add_argument("--tiles_embedding_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_embeddings_gigapath/224")
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import timm
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
    #tile_encoder = nn.DataParallel(tile_encoder)
    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    #channels = [0, 25, 28]
    #channels = [10, 11, 12]
    channels = [11, 24, 30]
    channels_directory_output = '_'.join(map(str, channels))
    tiles_embedding_path = os.path.join(tiles_embedding_path, channels_directory_output)
    batch_size = 128
    num_workers = 28
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
                                    for r, ds, fs in os.walk(os.path.join(cores_files_path,core))
                                    for fn in fs if fn.endswith('.tiff')])
        cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
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
        wholeslide_files = [files_path+slide_dir+'/tiles/'+file for slide_dir in os.listdir(files_path) for file in os.listdir(files_path+'/'+slide_dir+'/tiles') if
                                     os.path.isfile(files_path+slide_dir+'/tiles/'+file) and file.endswith('tiff')]
        wholeslide_labels_df = pd.read_csv('data/wholeslide_clinical_data.csv')
    tiff_dataset = TiffDataset(files=cores_files, files_names=cores_files, transform=transform, channels=channels)
    tile_encoder.eval()
    sampler = None
    loader = MultiEpochsDataLoader(
        tiff_dataset, batch_size=batch_size, shuffle=(sampler is None),
        pin_memory=True, sampler=sampler, num_workers=num_workers)
    existing_dirs = []
    for i, (images, file_names) in enumerate(loader):
        images = images.cuda()
        with torch.no_grad():
            output = tile_encoder(images)
        output = output.cpu()
        with Pool(num_workers) as pool:
           args = [(output[i], file_names[i], tiles_embedding_path) for i in range(len(output))
                   if not os.path.exists(os.path.join(tiles_embedding_path,
                                                      '/'.join(file_names[i].split('/')[-3:]).replace('.tiff',
                                                                                                      '') + '_tensor.pt'))
                   ]
           pool.starmap(save_embedding, args)
    #args = [(transform, image_path, tiles_embedding_path, tile_encoder) for image_path in files_train]
    #with Pool(32) as pool:
    #    pool.starmap(extract_embedding, args)

    #args = [(transform, image_path, tiles_embedding_path, tile_encoder) for image_path in files_test]
    #with Pool(32) as pool:
    #    pool.starmap(extract_embedding, args)
    # for file in files_train:
    #     sample_input = transform(Image.open(file).convert("RGB")).unsqueeze(0).cuda()
    #     pathlib.Path(pathlib.Path(os.path.join(tiles_embedding_path,'/'.join(file.split('/')[-3:-1])))).mkdir(parents=True, exist_ok=True)
    #     with torch.no_grad():
    #         output = tile_encoder(sample_input).squeeze()
    #         torch.save(output, os.path.join(tiles_embedding_path,'/'.join(file.split('/')[-3:]).replace('.tiff','')+'_tensor.pt'))
    # for file in files_test:
    #     sample_input = transform(Image.open(file).convert("RGB")).unsqueeze(0).cuda()
    #     pathlib.Path(pathlib.Path(os.path.join(tiles_embedding_path, '/'.join(file.split('/')[-3:-1])))).mkdir(
    #         parents=True, exist_ok=True)
    #     with torch.no_grad():
    #         output = tile_encoder(sample_input).squeeze()
    #         torch.save(output, os.path.join(tiles_embedding_path,'/'.join(file.split('/')[-3:]).replace('.tiff','')+'_tensor.pt'))


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
