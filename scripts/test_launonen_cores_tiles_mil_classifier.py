import torch
import torch.nn as nn
import csv
import gzip
import torch.utils
import torch.distributions
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms as T
import numpy as np
import pandas as pd
import time
import wandb
import os, sys, glob
from sklearn.metrics import f1_score, accuracy_score
import ipdb
import pathlib
import tifffile
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from os.path import dirname, join, abspath
import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
%load_ext autoreload
%autoreload 2
from utils import ProgressMeter, AverageMeter, save_checkpoint, MultiEpochsDataLoader, TensorDatasetMIL, collate_fn_MIL, remove_part_by_negative_index, get_percentiles_normalize, PercentileNormalize
from models.abmil import ABMIL
from models.amlab_mil import Attention, GatedAttention







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, defualt="model_best_embedding_nact_mil_fullcore_0_25_28_encoder_trainable_filterlowcancercells_False_imagenormalization_True.pth.tar")
    p = parser.parse_args()
    model_path = p.model_path
    batch_size = 1
    input_dimensions = None
    classifier_model = Attention(encoding='trainable', input_dim=input_dimensions)
    checkpoint = torch.load('../saved_models/{0}'.format(model_path))
    classifier_model.load_state_dict(checkpoint['state_dict'])
    classifier_model.eval()

    channels_launonen = [0, 30, 34]
    files_path = "/data/projects/pixel_project/datasets/Launonen_TMA/patches/histoprep_embeddings_uni/224/0_30_34"
    cores_directories = []
    slides_directories = ["/data/projects/pixel_project/datasets/Launonen_TMA/patches/histoprep_embeddings_uni/224/0_30_34/TMA1"]
    for i, slide in enumerate(slides_directories):
        for r, ds, fs in os.walk(slide):
            for dn in ds:
                if dn.startswith('core'):
                    patches_files = [file for file in os.listdir(os.path.join(r, dn))]
                    cores_directories.append(os.path.join(r, dn))

    raw_cores = [core.replace('histoprep_embeddings_uni','histoprep_generated').replace('/'+'_'.join(str(channel) for channel in channels_launonen),'') for core in cores_directories]
    percentile_1, percentile_99 = get_percentiles_normalize(raw_cores, channels_launonen)
    normalize_transform = PercentileNormalize(percentile_1, percentile_99)
    transforms_train = normalize_transform

    cores_test = []
    labels_test = []
    for i, core_directory in enumerate(cores_directories):
        cores_test.append(core_directory)
        labels_test.append(0)
    tensor_dataset_test = TensorDatasetMIL(slides=cores_test,transform=transforms_train, labels=labels_test, gigapath=False,multi_channels=None, channels=channels_launonen, raw_images=raw_cores)
    test_loader = MultiEpochsDataLoader(
                tensor_dataset_test, batch_size=1, shuffle=False,
                 pin_memory=True, sampler=None, num_workers=4, collate_fn=None)

    binary_predictions_list = []
    labels_predictions = []
    losses = AverageMeter('Loss', ':.20f')
    data_time = AverageMeter('Data', ':6.3f')
    classifier_model = classifier_model.cuda()
    wandb.init(project='pixel_ai', name="test_launonen_{0}".format(model_path), resume="allow")

    for i, (bag_tensor, raw_images, labels, files_names) in enumerate(test_loader):
        raw_images = raw_images.cuda()
        #mask = mask.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            loss, attention_weights = classifier_model.calculate_objective(raw_images, labels, eval_mode=True)
            error, binary_predictions = classifier_model.calculate_classification_error(raw_images, labels,eval_mode=True)

        binary_predictions = binary_predictions.cpu().tolist()
        # Compute loss

        losses.update(loss.item(), 1)
        binary_predictions_list.extend(binary_predictions)
        labels_predictions.extend(labels.cpu())
        #if i % 100 == 0:
        end = time.time()
        data_time.update(time.time() - end)
    accuracy_value = accuracy_score(labels_predictions, binary_predictions_list)
    f1_value = f1_score(labels_predictions, binary_predictions_list, average='macro')
    print(f1_value)
    print(accuracy_value)
    wandb.log({"accuracy": accuracy_value, "f1": f1_value,
               "loss": losses.avg})

if __name__ == "__main__":
    main()