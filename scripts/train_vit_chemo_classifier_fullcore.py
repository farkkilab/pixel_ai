import torch
import torch.nn as nn
import csv
import gzip
import torch.utils
import torch.distributions
import torchvision
from torchinfo import summary
from torchvision import datasets, transforms, models
import torchvision.transforms as T
import numpy as np
import pandas as pd
import time
import wandb
import os, sys, glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import ipdb
import pathlib
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from vit_pytorch import ViT
from vit_pytorch.recorder import Recorder
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader
import ipdb

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
        labels = torch.as_tensor(labels).cuda()
        data_time.update(time.time() - end)
        predictions = classifier_model(images)
        binary_predictions = torch.argmax(predictions,dim=1).cpu().tolist()
        # Compute loss
        loss = criterion(predictions, labels)
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
    #/data/projects/pixel_project/datasets/NKI_project_TMAs/
    #/data/projects/sciset/resized/
    parser.add_argument("--files_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="cores")


    p = parser.parse_args()
    files_path = p.files_path
    data_type = p.data_type
    if data_type == 'cores':
        cores_files = []
        cores_directories = [d for d in os.listdir(files_path) if
                             os.path.isdir(os.path.join(files_path, d)) and d.startswith('TMA')]
        for i, slide in enumerate(cores_directories):
            cores_files_path = str(files_path) + "/" + slide + "/Channels_all"
            cores_files.extend([os.path.join(r, fn)
                                for r, ds, fs in os.walk(cores_files_path)
                                for fn in fs if fn.endswith('.tif')])
        cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
        cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')
    elif data_type == 'whole_slide':
        wholeslide_files = [os.path.join(files_path, d) for d in os.listdir(files_path) if
                             os.path.isfile(os.path.join(files_path, d)) and d.endswith('tif')]
        wholeslide_labels_df = pd.read_csv('data/wholeslide_clinical_data.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # resized images, this channels correspond actually to 0, 12, 28
    #channels = [0, 1, 2]
    channels = [0, 12, 28]
    in_channels = len(channels)
    best_f1_test = 0
    input_dimensions = (1024, 1024)
    epochs = 100
    lr = 0.001
    num_workers = 0
    #num_workers = 28
    model_path = 'saved_models'
    model_name = "model_best_{1}_vit_{0}".format(str(channels),data_type)
    model = 'resnet'
    if model == 'vit':
        batch_size = 32
        classifier_model = ViT(
            image_size=input_dimensions[0],
            patch_size=128,
            num_classes=2,
            dim=2048,
            depth=12,
            heads=6,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)
    else:
        batch_size = 8
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        classifier_model = models.resnet50(pretrained=True)
        num_features = classifier_model.fc.in_features
        classifier_model.fc = nn.Linear(num_features, 2)

        # Move model to device
        classifier_model = classifier_model.to(device)
    #summary(classifier_model, input_size=(batch_size, in_channels, input_dimensions[0], input_dimensions[1]))
    config = {
        "learning_rate": lr,
        "architecture": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "channels": channels,
        "input_dimensions": input_dimensions,
        "vae_model":model_name
    }
    transform_to_image = T.ToPILImage()


    #checkpoint = torch.load('{}/{}_vae.pth.tar'.format(model_path, model_name))
    transforms_train = torch.nn.Sequential(
        T.CenterCrop(2048),
        T.RandomCrop(input_dimensions[0])
        #T.Resize([input_dimensions[0], input_dimensions[1]])
    )
    transforms_test = torch.nn.Sequential(
        T.CenterCrop(input_dimensions[0]),
        #T.Resize([input_dimensions[0], input_dimensions[1]])
    )

    classifier_model = nn.DataParallel(classifier_model)
    labels_train = []
    labels_test = []
    labels_validate = []
    # Only files that we have a label for
    files_train = []
    files_test = []
    files_validate = []
    # list to keep only patches with labels
    if data_type == 'cores':
        for i, core_file in enumerate(cores_files):
            patch_file_label_df = cores_chemo_labels_df[(cores_chemo_labels_df['cycif.slide']==core_file.split('/')[-3])&(cores_chemo_labels_df['cycif.core.id']==core_file.split('/')[-1].replace('.tif',''))]
            core_file_stats_row = cores_stats_df[(cores_stats_df['cycif.slide'] == core_file.split('/')[-3]) & (
                        cores_stats_df['cycif.core.id'] == core_file.split('/')[-1].replace('.tif', ''))]
            # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
            if not patch_file_label_df.empty and not core_file_stats_row.empty and str(patch_file_label_df.iloc[0]['therapy_sequence']).lower()!='na' and not pd.isnull(patch_file_label_df.iloc[0]['therapy_sequence']):

                if core_file_stats_row['N.cancer.cells'].iloc[0]>500:
                    if core_file.split('/')[-3]=='TMA_42_961':
                        files_test.append(core_file)
                        # If contains NACT, is a sample collected after chemotherapy exposure
                        if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                            labels_test.append(1)
                        else:
                            labels_test.append(0)
                    else:#if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                        files_train.append(core_file)
                        # If contains NACT, is a sample collected after chemotherapy exposure
                        if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                            labels_train.append(1)
                        else:
                            labels_train.append(0)
            else:
                print('Missing label for:'+core_file)
    elif data_type == 'whole_slide':
        for i, wholeslide_file in enumerate(wholeslide_files):
            wholeslide_file_label_df = wholeslide_labels_df[
                (
                            wholeslide_labels_df['imageid'] == 's'+wholeslide_file.split('/')[-1].replace('.ome.tif', '').replace('Sample_', ''))]
            if not wholeslide_file_label_df.empty  and str(
                    wholeslide_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                    wholeslide_file_label_df.iloc[0]['therapy_sequence']):


                if wholeslide_file.split('/')[-1] == 'Sample_06.ome.tif':
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
    if data_type == 'cores':
        config['total_patches'] = len(cores_files)
    else:
        config['total_patches'] = len(wholeslide_files)
    config['train_images'] = len(files_train)
    config['test_images'] = len(files_test)
    # , mode="disabled"
    wandb.init(project='pixel_ai', name="vit_chemo_classifier_fullcore", resume="allow", config=config)
    tiff_dataset_train = TiffDataset(files=files_train,transform=transforms_train, channels=channels,labels=labels_train)
    tiff_dataset_test = TiffDataset(files=files_test,transform=transforms_test, channels=channels,labels=labels_test)
    #tiff_dataset_validate = TiffDataset(files=cores_files_validate, transform=transforms, channels=channels,labels=cores_labels_validate)
    train_sampler = None
    train_loader = MultiEpochsDataLoader(
            tiff_dataset_train, batch_size=batch_size, shuffle=(train_sampler is None),
             pin_memory=True, sampler=train_sampler, num_workers=num_workers)
    test_sampler = None
    test_loader = MultiEpochsDataLoader(
            tiff_dataset_test, batch_size=batch_size, shuffle=(test_sampler is None),
             pin_memory=True, sampler=test_sampler, num_workers=num_workers)
    validate_sampler = None
    #validate_loader = torch.utils.data.DataLoader(
    #    tiff_dataset_validate, batch_size=batch_size, shuffle=(validate_sampler is None),
    #    pin_memory=True, sampler=validate_sampler, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
        }, is_best, 'vit_chemo_classifier_fullcore.pth.tar')
    test_image_input_list = []
    files_names = []
    #loss_validate, acc_validate, f1_validate = train_test(classifier_model, optimizer, validate_loader, epoch, train=False,
    #                                          criterion=criterion)
    #print("Accuracy validate: " + str(acc_validate))
    #print("F1 validate: " + str(f1_validate))
    #print('Loss validate ' + str(loss_validate))
    #classifier_model = Recorder(classifier_model)
    #for i, images in enumerate(test_loader):
    #    files_names.append(images[1])
    #    images = images[0].cuda()
    #    ipdb.set_trace()
    #    preds, attns = classifier_model(images)


if __name__ == "__main__":
    main()
