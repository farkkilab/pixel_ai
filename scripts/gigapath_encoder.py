import os

os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/projects/pixel_project/huggingface/cache"
import timm
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
from gigapath.pipeline import run_inference_with_tile_encoder
from gigapath.slide_encoder import create_model

slide_encoder = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
def main():
    parser = argparse.ArgumentParser()
    # wh "/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated"
    parser.add_argument("--files_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_generated")
    parser.add_argument("--tiles_embedding_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_embeddings")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="cores")
    p = parser.parse_args()
    files_path = p.files_path
    tiles_embedding_path = p.tiles_embedding_path
    data_type = p.data_type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).cuda()
    #tile_encoder = nn.DataParallel(tile_encoder)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    channels = [0, 12, 28]
    batch_size = 32
    num_workers = 28
    labels_train = []
    labels_test = []
    labels_validate = []
    # Only files that we have a label for
    files_train = []
    files_test = []
    files_validate = []

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
        cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')
    elif data_type == 'whole_slide':
        wholeslide_files = [files_path+slide_dir+'/tiles/'+file for slide_dir in os.listdir(files_path) for file in os.listdir(files_path+'/'+slide_dir+'/tiles') if
                                     os.path.isfile(files_path+slide_dir+'/tiles/'+file) and file.endswith('tiff')]
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

                if core_file.split('/')[-3] == 'TMA_42_961':
                    files_test.append(core_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                else:  # if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                    files_train.append(core_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
            else:
                print('Missing label for:' + core_file)
    elif data_type == 'whole_slide':
        for i, wholeslide_file in enumerate(wholeslide_files):
            wholeslide_file_label_df = wholeslide_labels_df[
                (
                        wholeslide_labels_df['imageid'] == 's' + wholeslide_file.split('Sample_')[-1].split('/')[0])]
            if not wholeslide_file_label_df.empty and str(
                    wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                wholeslide_file_label_df.iloc[0]['therapy_sequence']):

                if wholeslide_file.split('Sample_')[-1].split('/')[0] == '06':
                    files_test.append(wholeslide_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                else:  # if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                    files_train.append(wholeslide_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
            else:
                print('Missing label for:' + wholeslide_file)

    tiff_dataset_train = TiffDataset(files=files_train, files_names=files_train, transform=transform, channels=channels,
                                     labels=labels_train)
    tiff_dataset_test = TiffDataset(files=files_test, files_names=files_train, transform=transform, channels=channels, labels=labels_test)
    tile_encoder.eval()
    sampler = None
    train_loader = MultiEpochsDataLoader(
        tiff_dataset_train, batch_size=batch_size, shuffle=(sampler is None),
        pin_memory=True, sampler=sampler, num_workers=num_workers)
    test_loader = MultiEpochsDataLoader(
        tiff_dataset_test, batch_size=batch_size, shuffle=(sampler is None),
        pin_memory=True, sampler=sampler, num_workers=num_workers)
    existing_dirs = []
    for i, (images, file_names, labels) in enumerate(train_loader):
        images = images.cuda()
        with torch.no_grad():
            output = tile_encoder(images)
        output = output.cpu()
        with Pool(num_workers) as pool:
           args = [(output[i], file_names[i], tiles_embedding_path) for i in range(len(output))]
           pool.starmap(save_embedding, args)
    for i, (images, file_names, labels) in enumerate(test_loader):
        images = images.cuda()
        with torch.no_grad():
            output = tile_encoder(images)
        output = output.cpu()
        with Pool(num_workers) as pool:
           args = [(output[i], file_names[i], tiles_embedding_path) for i in range(len(output))]
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
    set_start_method('spawn')
    main()
