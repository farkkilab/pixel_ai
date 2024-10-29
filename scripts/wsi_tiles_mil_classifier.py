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
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from vit_pytorch import ViT
from utils import ProgressMeter, AverageMeter, save_checkpoint, MultiEpochsDataLoader, TensorDatasetMIL, collate_fn_MIL, remove_part_by_negative_index, get_percentiles_normalize, PercentileNormalize
from models.abmil import ABMIL
from models.amlab_mil import Attention, GatedAttention



def train_test(classifier_model, optimizer,loader, epoch,train, criterion, model_encoder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device("cuda")
    print(device)
    #device = torch.device("cuda:0")
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.20f')
    f1 = AverageMeter('f1', ':6.2f')
    acc = AverageMeter('accuracy', ':6.2f')
    end = time.time()
    if train:
        prefix = "Epoch: "
        classifier_model.train()
    else:
        prefix = "Test: "
        classifier_model.eval()

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=prefix + " [{}]".format(epoch))
    binary_predictions_list = []
    labels_predictions = []
    for i, item in enumerate(loader):
        optimizer.zero_grad()

        if model_encoder == 'trainable':
            raw_images, labels, files_names = item
            raw_images = raw_images.to(device)
            input_variable = raw_images
        else:
            labels, files_names = item
            #bag_tensor = bag_tensor.to(device)
            #input_variable = bag_tensor
        labels = labels.to(device)
        data_time.update(time.time() - end)
        #predictions, attention_weights = classifier_model(bag_tensor, mask)
        #predictions, attention_weights, A = classifier_model(bag_tensor)

        chunk_size = 8000
        patches_data = []
        for i_chunk in range(0, len(files_names), chunk_size):
            chunk = files_names[i_chunk:i_chunk + chunk_size]
            for patch_i in range(len(chunk)):
                patches_data.append(torch.load(chunk[patch_i][0]))
            if patches_data:
                input_variable = torch.stack(patches_data)
            else:
                input_variable = torch.zeros(len(chunk))
            input_variable = input_variable.to(device)
            if not train:
                with torch.no_grad():
                    loss, attention_weights = classifier_model.calculate_objective(input_variable, labels, eval_mode=True)
                    error, binary_predictions = classifier_model.calculate_classification_error(input_variable, labels,eval_mode=True)
            else:
                loss, attention_weights = classifier_model.calculate_objective(input_variable, labels)
                error, binary_predictions = classifier_model.calculate_classification_error(input_variable, labels)
            binary_predictions_list.extend(binary_predictions.cpu().tolist())
            labels_predictions.extend(labels.cpu())
            if train:
                loss.backward()
            losses.update(loss.item(), 1)
            batch_time.update(time.time() - end)
            progress.display(i_chunk)
        # Compute loss


        if train:
            #loss.backward()
            optimizer.step()



        #if i % 100 == 0:
        #progress.display(i)
    accuracy_value = accuracy_score(labels_predictions, binary_predictions_list)
    f1_value = f1_score(labels_predictions, binary_predictions_list, average='macro')
    return losses.avg, accuracy_value, f1_value


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.to(device)
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=Path,
                        default="/scratch/project_2003009/NKI_wsi_MHCII_histoprep_patches/")
    parser.add_argument("--architecture", type=str,
                        default="amlab_attention")
    parser.add_argument("--image_normalization", type=str,
                        default=False)
    #use more than 3 channels, look for other channel images of same core and slide
    parser.add_argument("--multi_channels", type=bool,
                        default=False)
    parser.add_argument("--model_encoder", type=str,
                        default="trainable")

    p = parser.parse_args()
    files_path = p.files_path
    architecture = p.architecture
    image_normalization = p.image_normalization
    if image_normalization == 'true':
        image_normalization = True
    else:
        image_normalization = False
    multi_channels = p.multi_channels
    model_encoder = p.model_encoder
    wsi_directories = []
    slides_directories = [d for d in os.listdir(files_path) if
                          os.path.isdir(os.path.join(files_path, d)) and d.startswith('S')]
    for i, slide in enumerate(slides_directories):
        slide_files_path = str(files_path) + "/" + slide
        for r, ds, fs in os.walk(slide_files_path):
            for dn in ds:
                wsi_directories.append(os.path.join(r, dn))
    wholeslide_labels_df = pd.read_csv('data/mhcii_data.csv')

    best_f1_test = 0
    epochs = 60
    lr = 0.0001
    num_workers = 28
    sampling_n = 5000
    #DNA1, CK7, Vimentin
    channels = [0, 2, 4]
    model_path = 'saved_models'
    if model_encoder == 'trainable':
        input_dimensions = 56
    elif model_encoder == 'uni':
        input_dimensions = (1024)
    elif model_encoder == 'gigapath':
        input_dimensions = (1536)
    if architecture == 'ABMIL':
        batch_size = 1
        classifier_model = ABMIL(input_dim=input_dimensions)
        criterion = nn.BCELoss()
    elif architecture == 'amlab_attention':
        batch_size = 1
        classifier_model = Attention(encoding=model_encoder, input_dim=input_dimensions)
        criterion = nn.BCELoss()
    elif architecture == 'amlab_gated_attention':
        batch_size = 1
        classifier_model = GatedAttention(input_dim=input_dimensions)
        criterion = nn.BCELoss()

    # Move model to device
    classifier_model = classifier_model.to(device)
    #summary(classifier_model, input_size=(batch_size, in_channels, input_dimensions[0], input_dimensions[1]))
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dimensions": input_dimensions,
        "model": architecture
    }

    transforms_train = None
    transforms_test = None
    #torch.distributed.init_process_group(backend = 'nccl', world_size = 2, init_method = '...')
    #classifier_model = nn.DataParallel(classifier_model)
    labels_train = []
    labels_test = []
    # Only files that we have a label for
    wsi_train = []
    wsi_test = []
    # list to keep only patches with labels

    for i, wsi_directory in enumerate(wsi_directories):
        patch_file_label_df = wholeslide_labels_df[(wholeslide_labels_df['imageid']==wsi_directory.split('/')[-2])]
        if not patch_file_label_df.empty and not pd.isnull(patch_file_label_df.iloc[0]['PFS']):
            if wsi_directory.split('/')[-2]=='S065_iOme' or wsi_directory.split('/')[-2]=='S072_iOme':
                wsi_test.append(wsi_directory)
                # If contains NACT, is a sample collected after chemotherapy exposure
                if str(patch_file_label_df.iloc[0]['PFS']).lower()=='long':
                    labels_test.append(1)
                else:
                    labels_test.append(0)
            else:
                wsi_train.append(wsi_directory)
                # If contains NACT, is a sample collected after chemotherapy exposure
                if str(patch_file_label_df.iloc[0]['PFS']).lower()=='long':
                    labels_train.append(1)
                else:
                    labels_train.append(0)
        else:
            print('Missing label for:'+wsi_directory)


    config['total_wsi'] = len(wsi_directories)
    config['train_images'] = len(wsi_train)
    config['test_images'] = len(wsi_test)
    # , mode="disabled"
    wandb.init(project='pixel_ai', name="wsimhcii_embedding_classifier_mil_tilescore_{0}_encoder_{1}_imagenormalization_{2}".format(str(files_path).split('/')[-1],model_encoder, image_normalization), resume="allow", config=config)
    if model_encoder == 'trainable':
        raw_cores_train = [core.replace('histoprep_embeddings_uni','NKI_histoprep_patches').replace('/'+'_'.join(str(channel) for channel in channels),'').replace('/10_11_12','') for core in wsi_train]
        raw_cores_test = [core.replace('histoprep_embeddings_uni', 'NKI_histoprep_patches').replace('/'+'_'.join(str(channel) for channel in channels), '').replace('/10_11_12','') for core
                           in wsi_test]
        if image_normalization:
            percentile_min, percentile_max, mean, std = get_percentiles_normalize(raw_cores_train+raw_cores_test, channels, min_percentil=1, max_percentil=99)
            print(percentile_min, percentile_max, mean, std)
            normalize_transform = PercentileNormalize(percentile_min, percentile_max, mean, std, normalization_strategy="min_max")
            transforms_train = normalize_transform
            transforms_test = normalize_transform
        else:
            #percentile_min = np.array([0]*len(channels))
            #percentile_max = np.array([65535] * len(channels))
            transforms_train = None
            transforms_test = None



    else:
        raw_cores_train = None
        raw_cores_test = None
        normalize_transform = None
    tensor_dataset_train = TensorDatasetMIL(slides=wsi_train,raw_images=raw_cores_train,transform=transforms_train,labels=labels_train, channels=channels, gigapath=False,multi_channels=multi_channels, image_normalization=image_normalization, resize_img=True, sampling=sampling_n)
    tensor_dataset_test = TensorDatasetMIL(slides=wsi_test,raw_images=raw_cores_test,transform=transforms_test, labels=labels_test, channels=channels, gigapath=False,multi_channels=multi_channels, image_normalization=image_normalization, resize_img=True, sampling=sampling_n)
    #tiff_dataset_validate = TiffDataset(files=cores_files_validate, transform=transforms, channels=channels,labels=cores_labels_validate)
    train_sampler = None
    train_loader = MultiEpochsDataLoader(
            tensor_dataset_train, batch_size=batch_size, shuffle=False,
             pin_memory=True, sampler=train_sampler, num_workers=num_workers, collate_fn=None)
    test_sampler = None
    test_loader = MultiEpochsDataLoader(
            tensor_dataset_test, batch_size=batch_size, shuffle=False,
             pin_memory=True, sampler=test_sampler, num_workers=num_workers, collate_fn=None)

    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(epochs):
        loss_train, acc_train, f1_train = train_test(classifier_model,optimizer, train_loader,epoch,train=True, criterion=criterion, model_encoder=model_encoder)
        loss_test, acc_test, f1_test = train_test(classifier_model,optimizer, test_loader, epoch, train=False, criterion=criterion, model_encoder=model_encoder)
        print("Accuracy training: "+str(acc_train))
        print("Accuracy test: "+str(acc_test))
        print("F1 training: " + str(f1_train))
        print("F1 test: " + str(f1_test))
        wandb.log({"loss_train": loss_train, "loss_test": loss_test,
                   "lr": optimizer.param_groups[0]['lr'],
                   "epoch": epoch,
                   "acc_train":acc_train,"acc_test":acc_test,
                   "f1_train":f1_train,"f1_test":f1_test})
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Loss test '+str(loss_test))
        is_best = f1_test > best_f1_test
        if is_best:
            best_f1_test = f1_test
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': classifier_model.state_dict(),
            'best_f1_test': best_f1_test,
            'optimizer': optimizer.state_dict(),
        }, is_best, 'wsimhcii_embedding_mil_fullcore_{0}_encoder_{1}_imagenormalization_{2}.pth.tar'.format(str(files_path).split('/')[-1],model_encoder, image_normalization))



if __name__ == "__main__":
    main()
