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
from sklearn.metrics import f1_score, accuracy_score
import ipdb
import pathlib
import argparse
from pathlib import Path
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from models.embed_subtype_classifier import EmbedSubtypeClassifier
from models.vanilla_vae import VanillaVAE
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset
import ipdb

def train_test(vae_model,classifier_model, optimizer,loader, epoch,train, criterion):
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
        images = images.cuda()
        labels = torch.as_tensor(labels).cuda()
        data_time.update(time.time() - end)
        x_hat = vae_model(images)
        z = x_hat[4]
        predictions = classifier_model(z)
        threshold = 0.5
        binary_predictions = [1 if prob >= threshold else 0 for prob in predictions.squeeze(1).cpu().tolist()]
        # Compute loss
        loss = criterion(predictions.squeeze(1), labels.float())
        #model.train()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), images.size(0))
        acc.update(accuracy_score(binary_predictions, labels.cpu()),images.size(0))
        f1.update(f1_score(labels.cpu(), binary_predictions, average='macro'), images.size(0))
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
        progress.display(i)
    return losses.avg, acc.avg, f1.avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cores_path", type=Path,
                        default="/data/projects/pixel_project/datasets/NKI_project_TMAs/")



    p = parser.parse_args()
    cores_path = p.cores_path
    cores_files = []
    cores_directories = [d for d in os.listdir(cores_path) if
                         os.path.isdir(os.path.join(cores_path, d)) and d.startswith('TMA')]
    for i, slide in enumerate(cores_directories):
        files_path = str(cores_path) + "/" + slide + "/Channels_all"
        cores_files.extend([os.path.join(r, fn)
                            for r, ds, fs in os.walk(files_path)
                            for fn in fs if fn.endswith('.tif')])
    cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    latent_dims = 16
    channels = [0, 1, 2]
    in_channels = len(channels)
    best_f1_test = 99999999
    hidden_dims = [ 16, 32, 64, 128, 256]
    input_dimensions = (1024, 1024)
    epochs = 100
    lr = 0.001
    model_path = 'saved_models'
    model_name = "model_best_allcores_fullcore_{0}".format(str(channels))
    config = {
        "learning_rate": lr,
        "architecture": "vanilla-VAE",
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dims": latent_dims,
        "hidden_dims": hidden_dims,
        "channels": channels,
        "input_dimensions": input_dimensions,
        "vae_model":model_name
    }
    transform_to_image = T.ToPILImage()


    vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dims, input_dimensions=input_dimensions, hidden_dims=hidden_dims).to(device) # GPU
    checkpoint = torch.load('{}/{}_vae.pth.tar'.format(model_path, model_name))


    for key in list(checkpoint['state_dict'].keys()):
        if 'module.' in key:
            checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    vae_model.load_state_dict(checkpoint['state_dict'])
    vae_model.eval()
    classifier_model = EmbedSubtypeClassifier(input_dim=16, output_dim=1).to(device)
    cores_labels_train = []
    cores_labels_test = []
    # Only files that we have a label for
    cores_files_train = []
    cores_files_test = []
    # list to keep only patches with labels
    for i, core_file in enumerate(cores_files):
        patch_file_label_df = cores_chemo_labels_df[(cores_chemo_labels_df['cycif.slide']==core_file.split('/')[-3])&(cores_chemo_labels_df['cycif.core.id']==core_file.split('/')[-1].replace('.tif',''))]
        if not patch_file_label_df.empty and str(patch_file_label_df.iloc[0]['therapy_sequence']).lower()!='na':
            if core_file.split('/')[-3]=='TMA_42_961':
                cores_files_test.append(core_file)
                # If contains NACT, is a sample collected after chemotherapy exposure
                if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                    cores_labels_test.append(0)
                else:
                    cores_labels_test.append(1)
            else:
                cores_files_train.append(core_file)
                # If contains NACT, is a sample collected after chemotherapy exposure
                if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                    cores_labels_train.append(0)
                else:
                    cores_labels_train.append(1)
        else:
            print('Missing label for:'+core_file)

    config['total_patches'] = len(cores_files)
    config['train_images'] = len(cores_files_train)
    config['test_images'] = len(cores_files_test)
    wandb.init(project='pixel_ai', name="embed_chemo_classifier_fullcore", resume="allow", config=config)
    tiff_dataset_train = TiffDataset(files=cores_files_train,transform=T.Resize([1024,1024]), channels=channels,labels=cores_labels_train)
    tiff_dataset_test = TiffDataset(files=cores_files_test,transform=T.Resize([1024,1024]), channels=channels,labels=cores_labels_test)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
            tiff_dataset_train, batch_size=batch_size, shuffle=(train_sampler is None),
             pin_memory=True, sampler=train_sampler, num_workers=64)
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
            tiff_dataset_test, batch_size=batch_size, shuffle=(test_sampler is None),
             pin_memory=True, sampler=test_sampler, num_workers=64)



    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epochs):
        loss_train, acc_train, f1_train = train_test(vae_model, classifier_model,optimizer, train_loader,epoch,train=True, criterion=criterion)
        loss_test, acc_test, f1_test = train_test(vae_model, classifier_model,optimizer, test_loader, epoch, train=False, criterion=criterion)
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
        is_best = f1_test < best_f1_test
        if is_best:
            best_f1_test = f1_test
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': classifier_model.state_dict(),
            'best_f1_test': best_f1_test,
            'optimizer': optimizer.state_dict(),
        }, is_best, 'embed_chemo_classifier_fullcore.pth.tar')

if __name__ == "__main__":
    main()
