import torch
import torch.nn as nn
import csv
import gzip
import torch.utils
import torch.distributions
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
import time
import wandb
import os, sys, glob
import ipdb
import pathlib
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from models.resnet_classifier import ResNet, ResNet50, RawSubtypeClassifier
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset


# set parameters
input_dimensions = (128, 128) #(sample_file.shape[0],sample_file.shape[1])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
latent_dims = 8
channels = [0, 1, 2]
in_channels = len(channels)
num_classes = 1
hidden_dims = [16, 32, 64, 128, 256]
epochs = 30
USE_GPU = True
lr = 0.001
best_loss_val = 99999999
transform_to_image = T.ToPILImage()

patches_labels_df = pd.read_csv('../data/patches_labels.csv')
patches_statistics_df = pd.read_csv('../data/patch_size_128_stat_channel0.csv')
cores_folder = 'TMA_18_810/'
files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
model_path='/data/projects/pixel_project/saved_models'
model_name = "model_best_allcores_randompatches_{0}".format(str(channels))
print("Done setting parameters!")


# load dataset
# FIXME create a Class to read in the patches.
patches_files = [os.path.join(r, fn)
        for r, ds, fs in os.walk(files_path)
        for fn in fs if fn.endswith('.tiff')]
patches_statistics_df.sort_values(by=['Median'], ascending=False, inplace=True)
patches_statistics_df = patches_statistics_df[patches_statistics_df['Core']!='core124']
highest_median_pathes = ['/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/TMA_18_810/{0}/{1}'.format(row['Core'],row['Patch']) for i, row in patches_statistics_df.head(20000).iterrows()]
patches_files = [file for file in patches_files if file in highest_median_pathes]
patches_labels = []

patches_files_filtered = []
for i, patch_file in enumerate(patches_files): # list to keep only patches with labels
    patch_file_label_df = patches_labels_df[patches_labels_df['patch']==patch_file]
    if not patch_file_label_df.empty:
        if patch_file_label_df.iloc[0]['label'] == 'BRCAmut/met':
            patches_labels.append(0)
        elif patch_file_label_df.iloc[0]['label'] == 'HRP':
            patches_labels.append(1)
        patches_files_filtered.append(patch_file)

indices_train, indices_test  = train_test_split(range(len(patches_files_filtered)), test_size=0.1, random_state=42)
patches_files_filtered_train = [patches_files_filtered[i] for i in indices_train]
patches_files_filtered_test = [patches_files_filtered[i] for i in indices_test]

patches_labels_train = [patches_labels[i] for i in indices_train]
patches_labels_test = [patches_labels[i] for i in indices_test]

tiff_dataset_train = TiffDataset(files=patches_files_filtered_train, channels=channels,labels=patches_labels_train)
tiff_dataset_test = TiffDataset(files=patches_files_filtered_test,channels=channels,labels=patches_labels_test)

train_sampler = None
train_loader = torch.utils.data.DataLoader(
        tiff_dataset_train, batch_size=batch_size, shuffle=(train_sampler is None),
         pin_memory=True, sampler=train_sampler)
test_sampler = None
test_loader = torch.utils.data.DataLoader(
        tiff_dataset_test, batch_size=batch_size, shuffle=(test_sampler is None),
         pin_memory=True, sampler=test_sampler)
print("Done setting data loader!")


# define models
resnet50_model = torchvision.models.resnet50(pretrained=False).to(device)
for name, param in resnet50_model.named_parameters(): # freeze the parameters of the network, is this the correct place? 
    if 'fc' not in name:
        print(name, param.requires_grad)
        param.requires_grad=False
in_features = resnet50_model.fc.in_features
resnet50_model.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=in_features,
        out_features=num_classes
    ),
    torch.nn.Sigmoid()
) # change the number of classes in the final Linear layer
resnet50_model.cuda()
print("Done setting ResNet50 model!")

# define training function
# ref: https://www.kaggle.com/code/gxkok21/resnet50-with-pytorch
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(resnet50_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_test(model, optimizer, loader, epoch, train):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.20f')
    f1 = AverageMeter('f1', ':6.2f')
    acc = AverageMeter('accuracy', ':6.2f')
    end = time.time()

    if train:
        prefix = "Epoch: "
        model.train()
    else:
        prefix = "Test: "
        model.eval()

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=prefix + " [{}]".format(epoch))

    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        y = model(images)
        loss = criterion(y.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        threshold = 0.5
        binary_predictions = [1 if prob >= threshold else 0 for prob in y.squeeze(1).cpu().tolist()]
        losses.update(loss.item(), images.size(0))
        acc.update(accuracy_score(binary_predictions, labels.cpu()),images.size(0))
        f1.update(f1_score(labels.cpu(), binary_predictions, average='macro'), images.size(0))
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
        progress.display(i)
    return losses.avg, acc.avg, f1.avg

# ipdb.set_trace()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="resnet-core-classifier",
    # we can set run_names (for the future tests)
    # track hyperparameters and run metadata
    config={
    "architecture": "ResNet50",
    "dataset": "TMA_18_810",
    "epochs": 30,
    }
)

for epoch in range(epochs):
    loss_train, acc_train, f1_train = train_test(resnet50_model, optimizer, train_loader, epoch, train=True)
    loss_test, acc_test, f1_test = train_test(resnet50_model, optimizer, test_loader, epoch, train=False)
    print("Accuracy training: "+str(acc_train))
    print("Accuracy test: "+str(acc_test))
    wandb.log({"loss_train": loss_train, "loss_test": loss_test,
              "Accuracy training": acc_train, "Accuracy test": acc_test, "lr": optimizer.param_groups[0]['lr'],
              "epoch": epoch})
    
    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    print('Loss test '+str(loss_test))
    # is_best = loss_test < best_loss_val
    # if is_best:
    #    best_loss_val = loss_test
    # save_checkpoint({
    #    'epoch': epoch + 1,
    #    'state_dict': resnet50_model.state_dict(),
    #    'best_loss_val': best_loss_val,
    #    'optimizer': optimizer.state_dict(),
    # }, is_best, '{0}_resnet.pth.tar'.format(model_name))





