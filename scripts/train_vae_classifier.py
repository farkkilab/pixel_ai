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
import itertools
import ipdb
import pathlib
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from models.embed_subtype_classifier import EmbedSubtypeClassifier
from models.vanilla_vae import VanillaVAE
from utils.utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader


def train_test(vae_model,classifier_model, optimizer,loader, epoch,train):
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
        labels = labels.cuda()
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
    cores_folder = 'TMA_18_810/'
    files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
    input_dimensions = (1024, 1024)#(sample_file.shape[0],sample_file.shape[1])
    patches_labels_df = pd.read_csv('data/patches_labels.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    num_workers = 28
    latent_dims = 32
    channels = [0, 25, 28]
    in_channels = len(channels)
    hidden_dims = [ 16, 32, 64, 128, 256]
    patches_statistics_df = pd.read_csv('data/patch_size_128_stat_channel0.csv')
    input_dimensions = (128, 128)
    epochs = 30
    lr = 0.001
    transform = T.Compose([
        T.CenterCrop(1024),
        T.Resize([1024, 1024]),
    ])
    cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')

    model_path='saved_models'
    model_name = "model_best_allcores_vaeclassifier"

    vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dims, input_dimensions=input_dimensions, hidden_dims=hidden_dims).to(device) # GPU
    checkpoint = torch.load('{}/{}_vae.pth.tar'.format(model_path, model_name))


    for key in list(checkpoint['state_dict'].keys()):
        if 'module.' in key:
            checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    vae_model.load_state_dict(checkpoint['state_dict'])
    vae_model.eval()
    classifier_model = EmbedSubtypeClassifier(input_dim=8, output_dim=1).to(device)
    labels_train = []
    labels_test = []
    labels_validate = []
    # Only files that we have a label for
    cores_train = []
    cores_test = []
    cores_validate = []
    # list to keep only patches with labels
    cores_directories = []
    slides_directories = [d for d in os.listdir(files_path) if
                          os.path.isdir(os.path.join(files_path, d)) and d.startswith('TMA')]
    slide_combinations = list(itertools.combinations(slides_directories, 2))
    limited_combinations = slide_combinations[:10]
    best_fold_accuracies = []
    best_fold_f1_scores = []
    min_fold_test_losses = []

    for fold_num, test_slides in enumerate(limited_combinations):

        # Initialize lists for core directories for this fold
        cores_train_directory = []
        cores_test_directory = []

        # Loop over slides to collect core directories based on current fold's split
        for slide in slides_directories:
            slide_files_path = os.path.join(files_path, slide) + "/Channels_all/"
            for r, ds, fs in os.walk(slide_files_path):
                for fn in fs:
                    if fn.startswith('core'):
                        core_path = os.path.join(r, fn)
                        cores_directories.append(core_path)
                        # Assign core to train or test based on whether slide is in test_slides
                        if slide in test_slides:
                            cores_test_directory.append(core_path)
                        else:
                            cores_train_directory.append(core_path)
        # patches_files = [file for file in patches_files if file in highest_median_pathes]
        for i, core_directory in enumerate(cores_directories):
            patch_file_label_df = cores_chemo_labels_df[
                (cores_chemo_labels_df['cycif.slide'] == core_directory.split('/')[-3]) & (
                        cores_chemo_labels_df['cycif.core.id'] == core_directory.split('/')[-1].replace(".tif", ""))]
            # if filter_cores_lowcancer:
            #    core_file_stats_row = cores_stats_df[(cores_stats_df['cycif.slide'] == core_directory.split('/')[-2]) & (
            #            cores_stats_df['cycif.core.id'] == core_directory.split('/')[-1])]

            # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
            # Ignore labels with PDS followed by NACT
            if not patch_file_label_df.empty and str(
                    patch_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                patch_file_label_df.iloc[0]['therapy_sequence']) \
                    and not patch_file_label_df.iloc[0]['therapy_sequence'] == 'PDS followed by NACT':
                # skip the rows if filter_cores_lowcancer is set to true and the core don't have enough cancer cells
                # if filter_cores_lowcancer and (
                #        core_file_stats_row.empty or core_file_stats_row['N.cancer.cells'].iloc[0] < 500):
                # +    continue
                if core_directory in cores_test_directory:
                    cores_test.append(core_directory)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                else:
                    cores_train.append(core_directory)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
            else:
                print('Missing label for:' + core_directory)
        tiff_dataset_train = TiffDataset(files=cores_train_directory, channels=channels, labels=labels_train,
                                         transform=transform)
        tiff_dataset_test = TiffDataset(files=cores_test_directory, channels=channels, labels=labels_test,
                                        transform=transform)

        train_sampler = None
        train_loader = MultiEpochsDataLoader(
            tiff_dataset_train, batch_size=batch_size, shuffle=False,
            pin_memory=True, sampler=train_sampler, num_workers=num_workers, collate_fn=None)
        test_sampler = None
        test_loader = MultiEpochsDataLoader(
            tiff_dataset_test, batch_size=batch_size, shuffle=False,
            pin_memory=True, sampler=test_sampler, num_workers=num_workers, collate_fn=None)
        print("Done setting data loader!")

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        for epoch in range(epochs):
            loss_train, acc_train, f1_train = train_test(vae_model, classifier_model,optimizer, train_loader,epoch,train=True)
            loss_test, acc_test, f1_test = train_test(vae_model, classifier_model,optimizer, test_loader, epoch, train=False)
            print("Accuracy training: "+str(acc_train))
            print("Accuracy test: "+str(acc_test))
            #wandb.log({"loss_train": loss_train, "loss_test": loss_test,
            #           "lr": optimizer.param_groups[0]['lr'],
            #           "epoch": epoch})
            scheduler.step()
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            print('Loss test '+str(loss_test))
            #is_best = loss_test < best_loss_val
            #if is_best:
            #    best_loss_val = loss_test
            #save_checkpoint({
            #    'epoch': epoch + 1,
            #    'state_dict': model.state_dict(),
            #    'best_loss_val': best_loss_val,
            #    'optimizer': optimizer.state_dict(),
            #}, is_best, '{0}_vae.pth.tar'.format(model_name))