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
import os, sys, glob
from sklearn.model_selection import train_test_split
import ipdb
import pathlib
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from models.vanilla_vae import VanillaVAE
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset




device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 30
batch_size = 4096
lr = 0.0001
latent_dims = 32
hidden_dims = [32, 64, 128, 256, 512]
best_loss_val = 99999999
transform_to_image = T.ToPILImage()
cores_folder = 'TMA_18_810/'
files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
model_name = "allcores_randompatches"
input_dimensions = (32, 32)#(sample_file.shape[0],sample_file.shape[1])
def train_test(model, optimizer,loader, epoch,train):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.20f')
    end = time.time()
    if train:
        prefix = "Epoch: "
    else:
        prefix = "Test: "
        model.eval()


    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=prefix + " [{}]".format(epoch))

    for i, images in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        x_hat = model(images)
        loss = model.module.loss_function(*x_hat,
                                              M_N=0.0008)
        model.train()
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()



        losses.update(loss['loss'].item(), images.size(0))
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
    progress.display(i)
    return losses.avg

#pathlib.Path("saved_models/{0}/images".format(cores_folder)).mkdir(parents=True, exist_ok=True)
patches_files = [os.path.join(r, fn)
        for r, ds, fs in os.walk(files_path)
        for fn in fs if fn.endswith('.tiff')]
patches_files_train, patches_files_test = train_test_split(patches_files, test_size=0.1, random_state=42)
#matrix_files_train = [file for file in patches_files_train['full_path_file_name'].to_list() if file.endswith(".npy")]
#matrix_files_test = [file for file in patches_files_test['full_path_file_name'].to_list() if file.endswith(".npy")]

model = VanillaVAE(in_channels=38,latent_dim=latent_dims,input_dimensions=input_dimensions,hidden_dims=hidden_dims).to(device) # GPU
model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
tiff_dataset_train = TiffDataset(files=patches_files_train,transform=T.Resize(input_dimensions))
tiff_dataset_test = TiffDataset(files=patches_files_test,transform=T.Resize(input_dimensions))

train_sampler = None
train_loader = torch.utils.data.DataLoader(
        tiff_dataset_train, batch_size=batch_size, shuffle=(train_sampler is None),
         pin_memory=True, sampler=train_sampler)
test_sampler = None
test_loader = torch.utils.data.DataLoader(
        tiff_dataset_test, batch_size=batch_size, shuffle=(test_sampler is None),
         pin_memory=True, sampler=test_sampler)
for epoch in range(epochs):
    loss_train = train_test(model,optimizer, train_loader,epoch,train=True)
    loss_test = train_test(model,optimizer, test_loader, epoch, train=False)
    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    print('Loss test '+str(loss_test))
    is_best = loss_test < best_loss_val
    if is_best:
        best_loss_val = loss_test
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss_val': best_loss_val,
        'optimizer': optimizer.state_dict(),
    }, is_best, '{0}_vae.pth.tar'.format(model_name))
