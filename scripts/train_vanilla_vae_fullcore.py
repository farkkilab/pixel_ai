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
from sklearn.model_selection import train_test_split
import ipdb
import pathlib
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from models.vanilla_vae import VanillaVAE
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset




def train_test(model, optimizer,loader, epoch,train, kld_weight):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.20f')
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

    for i, images in enumerate(loader):
        data_time.update(time.time() - end)
        images = images[0].cuda()
        x_hat = model(images)
        loss = model.module.loss_function(*x_hat,
                                              M_N=kld_weight)
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()



        losses.update(loss['loss'].item(), images.size(0))
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
    progress.display(i)
    return losses.avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cores_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/")
    parser.add_argument("--latent_dims", type=int,
                        default=8)
    parser.add_argument("--lr", type=float,
                        default=0.0001)


    p = parser.parse_args()
    cores_path = p.cores_path
    latent_dims = p.latent_dims
    lr = p.lr
    #pathlib.Path("saved_models/{0}/images".format(cores_folder)).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 250
    batch_size = 32
    channels = [0, 1, 2]
    # channels = [0, 1, 2, 25, 27, 29]
    in_channels = len(channels)
    hidden_dims = [16, 32, 64, 128, 256]
    best_loss_val = 99999999
    kld_weight = 0.000025
    transform_to_image = T.ToPILImage()
    model_name = "allcores_fullcore_{0}".format(str(channels))
    input_dimensions = (1024, 1024)  # (sample_file.shape[0],sample_file.shape[1])
    config = {
        "learning_rate": lr,
        "architecture": "vanilla-VAE",
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dims": latent_dims,
        "hidden_dims": hidden_dims,
        "channels": channels,
        "input_dimensions": input_dimensions,
        "kld_weight": kld_weight
    }
    cores_files = []
    cores_directories = [d for d in os.listdir(cores_path) if
                           os.path.isdir(os.path.join(cores_path, d)) and d.startswith('TMA')]
    for i, slide in enumerate(cores_directories):
        files_path = str(cores_path) + "/" + slide + "/Channels_all"
        cores_files.extend([os.path.join(r, fn)
                              for r, ds, fs in os.walk(files_path)
                              for fn in fs if fn.endswith('.tif')])
    cores_files_train, cores_files_test = train_test_split(cores_files, test_size=0.1, random_state=42)
    config['total_patches'] = len(cores_files)
    config['train_images'] = len(cores_files_train)
    config['test_images'] = len(cores_files_test)
    wandb.init(project='pixel_ai', name=model_name, resume="allow", config=config)
    #matrix_files_train = [file for file in patches_files_train['full_path_file_name'].to_list() if file.endswith(".npy")]
    #matrix_files_test = [file for file in patches_files_test['full_path_file_name'].to_list() if file.endswith(".npy")]

    model = VanillaVAE(in_channels=in_channels,latent_dim=latent_dims,input_dimensions=input_dimensions,hidden_dims=hidden_dims).to(device) # GPU
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    tiff_dataset_train = TiffDataset(files=cores_files_train,transform=T.Resize([1024,1024]),files_names=cores_files_train, channels=channels)
    tiff_dataset_test = TiffDataset(files=cores_files_test,transform=T.Resize([1024,1024]),files_names=cores_files_test, channels=channels)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
            tiff_dataset_train, batch_size=batch_size, shuffle=(train_sampler is None),
             pin_memory=True, sampler=train_sampler, num_workers=64)
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
            tiff_dataset_test, batch_size=batch_size, shuffle=(test_sampler is None),
             pin_memory=True, sampler=test_sampler, num_workers=64)
    for epoch in range(epochs):
        loss_train = train_test(model,optimizer, train_loader,epoch,train=True,kld_weight=kld_weight)
        loss_test = train_test(model,optimizer, test_loader, epoch, train=False,kld_weight=kld_weight)
        wandb.log({"loss_train": loss_train, "loss_test": loss_test,
                   "lr": optimizer.param_groups[0]['lr'],
                   "epoch": epoch})
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
    torch.cuda.empty_cache()
    train_image_input_list = []
    train_image_reconstructed_list = []
    files_names = []
    for i, images in enumerate(train_loader):
        files_names.append(images[1])
        images = images[0].cuda()
        x_hat = model(images)
        x_hat[0] = x_hat[0].cpu()
        for i_batch, test_image in enumerate(x_hat[0]):
            img_reconstructed = transform_to_image(test_image[0,:,:])
            train_image_reconstructed_list.append(img_reconstructed)
            if i_batch >10:
                break
        x_hat[1] = x_hat[1].cpu()
        for i_batch, test_image in enumerate(x_hat[1]):
            img_input = transform_to_image(test_image[0,:,:])
            train_image_input_list.append(img_input)
            if i_batch >10:
                break
    wandb.log({"train_image_input_list": [wandb.Image(image) for image in train_image_input_list],
               "train_image_reconstructed_list": [wandb.Image(image) for image in train_image_reconstructed_list]})
if __name__ == "__main__":
    main()
