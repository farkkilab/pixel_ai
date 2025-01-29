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
from utils import ProgressMeter, AverageMeter, save_checkpoint, TensorDataset, MultiEpochsDataLoader, TensorDataset2D
from models.embed_subtype_classifier_2d import EmbedSubtypeClassifier2D

def train_test(classifier_model, optimizer,loader, epoch,train, criterion):
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
    for i, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        images = images.cuda()
        if classifier_model.__class__.__name__ == 'ResNet':
            images = images[:,None,:,:, :]
            images = images.repeat(1,3,1,1,1)
            images = images.view(images.shape[0], 3 * 1024, 6, 6)

        labels = torch.as_tensor(labels).cuda()
        data_time.update(time.time() - end)
        predictions = classifier_model(images)
        if classifier_model.__class__.__name__ == 'ResNet':
            binary_predictions = torch.argmax(predictions,dim=1).cpu().tolist()
            # Compute loss
            loss = criterion(predictions, labels)
        else:
            binary_predictions = (predictions > 0.5).float().cpu().tolist()
            # Compute loss
            loss = criterion(predictions.squeeze(), labels.float())

        if train:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        acc.update(accuracy_score(binary_predictions, labels.cpu()),images.size(0))
        f1.update(f1_score(labels.cpu(), binary_predictions, average='macro'), images.size(0))
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
        progress.display(i)
    return losses.avg, acc.avg, f1.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/histoprep_embeddings_uni/256/10_11_12")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="cores")
    parser.add_argument("--classification_task", type=str,
                        default="nact")


    p = parser.parse_args()
    files_path = p.files_path
    data_type = p.data_type
    classification_task = p.classification_task
    if data_type == 'cores':
        cores_directories = []
        slides_directories = [d for d in os.listdir(files_path) if
                             os.path.isdir(os.path.join(files_path, d)) and d.startswith('TMA')]
        for i, slide in enumerate(slides_directories):
            slide_files_path = str(files_path) + "/" + slide
            for r, ds, fs in os.walk(slide_files_path):
                for dn in ds:
                    if dn.startswith('core'):
                        patches_files = [file for file in os.listdir(os.path.join(r, dn))]
                        cores_directories.append(os.path.join(r, dn))
        cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
    elif data_type == 'whole_slide':
        wholeslide_files = [os.path.join(files_path, d) for d in os.listdir(files_path) if
                             os.path.isfile(os.path.join(files_path, d)) and d.endswith('tif')]
        wholeslide_labels_df = pd.read_csv('data/wholeslide_clinical_data.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_f1_test = 0
    input_dimensions = (8, 8)
    epochs = 120
    lr = 0.0001
    num_workers = 28
    model_path = 'saved_models'
    model = 'custom_embed_classifier'
    if model == 'custom_embed_classifier':
        batch_size = 512
        classifier_model = EmbedSubtypeClassifier2D(input_dim=input_dimensions, output_dim=1)
        criterion = nn.BCELoss()
    elif model == 'resnet':
        batch_size = 256
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        classifier_model = models.resnet50(pretrained=True)
        classifier_model.conv1 = nn.Conv2d(3 * 1024, 64, kernel_size=7, stride=2, padding=3, bias=False)
        criterion = nn.CrossEntropyLoss()

        num_features = classifier_model.fc.in_features
        classifier_model.fc = nn.Linear(num_features, 2)

    # Move model to device
    classifier_model = classifier_model.to(device)
    #summary(classifier_model, input_size=(batch_size, in_channels, input_dimensions[0], input_dimensions[1]))
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dimensions": input_dimensions,
        "model": model
    }

    transforms_train = None
    transforms_test = None
    #torch.distributed.init_process_group(backend = 'nccl', world_size = 2, init_method = '...')
    #classifier_model = nn.DataParallel(classifier_model)
    labels_train = []
    labels_test = []
    labels_validate = []
    # Only files that we have a label for
    cores_train = []
    cores_test = []
    cores_validate = []
    # list to keep only patches with labels
    if data_type == 'cores':
        for i, core_directory in enumerate(cores_directories):
            patch_file_label_df = cores_chemo_labels_df[(cores_chemo_labels_df['cycif.slide']==core_directory.split('/')[-2])&(cores_chemo_labels_df['cycif.core.id']==core_directory.split('/')[-1])]
            if classification_task == "nact":
                # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
                # Ignore labels with PDS followed by NACT
                if not patch_file_label_df.empty and str(patch_file_label_df.iloc[0]['therapy_sequence']).lower()!='na' and not pd.isnull(patch_file_label_df.iloc[0]['therapy_sequence'])\
                        and not patch_file_label_df.iloc[0]['therapy_sequence'] == 'PDS followed by NACT':
                    # ignore images with nan
                    if core_directory.split('/')[-2]=='TMA_41_812':
                        cores_test.append(core_directory)
                        # If contains NACT, is a sample collected after chemotherapy exposure
                        if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                            labels_test.append(1)
                        else:
                            labels_test.append(0)
                    else:#if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                        cores_train.append(core_directory)
                        # If contains NACT, is a sample collected after chemotherapy exposure
                        if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                            labels_train.append(1)
                        else:
                            labels_train.append(0)
                else:
                    print('Missing label for:'+core_directory)
            elif classification_task == "progression":
                # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
                if not patch_file_label_df.empty and str(
                        patch_file_label_df.iloc[0]['progression']).lower() != 'na' and not pd.isnull(
                        patch_file_label_df.iloc[0]['progression']):
                    # ignore images with nan
                    if core_directory.split('/')[-2] == 'TMA_42_961':
                        cores_test.append(core_directory)

                        labels_test.append(patch_file_label_df.iloc[0]['progression'])
                    else:  # if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                        cores_train.append(core_directory)
                        # If contains NACT, is a sample collected after chemotherapy exposure
                        labels_train.append(patch_file_label_df.iloc[0]['progression'])
    elif data_type == 'whole_slide':
        for i, wholeslide_file in enumerate(wholeslide_files):
            wholeslide_file_label_df = wholeslide_labels_df[
                (
                            wholeslide_labels_df['imageid'] == 's'+wholeslide_file.split('/')[-1].replace('.ome.tif', '').replace('Sample_', ''))]
            if not wholeslide_file_label_df.empty  and str(
                    wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                    wholeslide_file_label_df.iloc[0]['therapy_sequence']):


                if wholeslide_file.split('/')[-1] == 'Sample_06.ome.tif':
                    cores_test.append(wholeslide_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                else:  # if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                    cores_train.append(wholeslide_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
            else:
                print('Missing label for:' + wholeslide_file)
    if data_type == 'cores':
        config['total_cores'] = len(cores_directories)
    else:
        config['total_patches'] = len(wholeslide_files)
    config['train_images'] = len(cores_train)
    config['test_images'] = len(cores_test)
    # , mode="disabled"
    wandb.init(project='pixel_ai', name="embedding_chemo_classifier_tilescore", resume="allow", config=config)
    tensor_dataset_train = TensorDataset2D(slides=cores_train,transform=transforms_train,labels=labels_train, gigapath=False)
    tensor_dataset_test = TensorDataset2D(slides=cores_test,transform=transforms_test, labels=labels_test, gigapath=False)
    #tiff_dataset_validate = TiffDataset(files=cores_files_validate, transform=transforms, channels=channels,labels=cores_labels_validate)
    train_sampler = None
    train_loader = MultiEpochsDataLoader(
            tensor_dataset_train, batch_size=batch_size, shuffle=False,
             pin_memory=True, sampler=train_sampler, num_workers=num_workers)
    test_sampler = None
    test_loader = MultiEpochsDataLoader(
            tensor_dataset_test, batch_size=batch_size, shuffle=False,
             pin_memory=True, sampler=test_sampler, num_workers=num_workers)

    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(epochs):
        loss_train, acc_train, f1_train = train_test(classifier_model,optimizer, train_loader,epoch,train=True, criterion=criterion)
        loss_test, acc_test, f1_test = train_test(classifier_model,optimizer, test_loader, epoch, train=False, criterion=criterion)

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
        }, is_best, 'embedding_chemo_classifier_fullcore.pth.tar')



if __name__ == "__main__":
    main()
