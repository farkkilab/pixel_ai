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
import argparse
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
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader


def train_test(vae_model, classifier_model, optimizer, loader, epoch, train, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.20f')
    f1 = AverageMeter('f1', ':6.2f')
    acc = AverageMeter('accuracy', ':6.2f')
    end = time.time()

    if train:
        prefix = "Epoch: "
        classifier_model.train()
        vae_model.eval()  # Keep VAE in eval mode during training
    else:
        prefix = "Test: "
        classifier_model.eval()
        vae_model.eval()

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=prefix + " [{}]".format(epoch))

    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        data_time.update(time.time() - end)
        # Disable gradient computation for VAE
        with torch.no_grad():
            x_hat = vae_model(images)
            z = x_hat[4]

        # Zero gradients for classifier
        optimizer.zero_grad()

        # Forward pass and loss computation for classifier
        predictions = classifier_model(z)
        threshold = 0.5
        binary_predictions = [1 if prob >= threshold else 0 for prob in predictions.squeeze(1).cpu().tolist()]

        # Compute loss
        loss = criterion(predictions.squeeze(1), labels.float())

        if train:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        acc.update(accuracy_score(binary_predictions, labels.cpu()), images.size(0))
        f1.update(f1_score(labels.cpu(), binary_predictions, average='macro'), images.size(0))
        batch_time.update(time.time() - end)
        progress.display(i)

    return losses.avg, acc.avg, f1.avg


def main():
    # Add an argument parser for filter_cores_lowcancer
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_cores_lowcancer", type=str, default=False)
    p = parser.parse_args()
    filter_cores_lowcancer = p.filter_cores_lowcancer == 'true'

    # Set parameters
    input_dimensions = (1024, 1024)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    num_workers = 28
    latent_dims = 16
    channels = [0, 25, 28]
    in_channels = len(channels)
    hidden_dims = [16, 32, 64, 128, 256]
    input_dimensions = (1024, 1024)
    epochs = 30
    lr = 0.001
    transform = T.Compose([
        T.CenterCrop(1536),
        T.Resize([1024, 1024]),
    ])

    cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')

    # Optional: Load cores_stats_df if filter_cores_lowcancer is used
    if filter_cores_lowcancer:
        cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')

    model_path = 'saved_models'
    model_name = "model_best_vae_allcores_randomcores_[0, 25, 28]_vae"

    # Load pre-trained VAE model
    vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dims, input_dimensions=input_dimensions,
                           hidden_dims=hidden_dims).to(device)
    checkpoint = torch.load('{}/{}.pth.tar'.format(model_path, model_name))

    # Clean up checkpoint keys if needed
    for key in list(checkpoint['state_dict'].keys()):
        if 'module.' in key:
            checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    vae_model.load_state_dict(checkpoint['state_dict'])
    vae_model.eval()

    # Prepare for cross-validation
    classifier_model = EmbedSubtypeClassifier(input_dim=16, output_dim=1).to(device)
    cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
    cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')
    files_path = '/scratch/project_2003009/NKI_project_TMAs'
    cores_stats_df.sort_values(by=['N.cancer.cells'], ascending=False, inplace=True)
    # highest_median_pathes = [str(cores_path) + '/{0}/{1}/{2}'.format(row['Slide'], row['Core'], row['Patch']) for
    #                         i, row in cores_statistics_df[cores_statistics_df['N.cancer.cells'] > 500].iterrows()]

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
    limited_combinations = slide_combinations[:5]
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
            if filter_cores_lowcancer:
                core_file_stats_row = cores_stats_df[
                    (cores_stats_df['cycif.slide'] == core_directory.split('/')[-3]) & (
                            cores_stats_df['cycif.core.id'] == core_directory.split('/')[-1].replace(".tif", ""))]
            # if filter_cores_lowcancer:
            #    core_file_stats_row = cores_stats_df[(cores_stats_df['cycif.slide'] == core_directory.split('/')[-2]) & (
            #            cores_stats_df['cycif.core.id'] == core_directory.split('/')[-1])]

            # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
            # Ignore labels with PDS followed by NACT
            if not patch_file_label_df.empty and str(
                    patch_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                patch_file_label_df.iloc[0]['therapy_sequence']) \
                    and not patch_file_label_df.iloc[0]['therapy_sequence'] == 'PDS followed by NACT':
                if filter_cores_lowcancer and (
                        core_file_stats_row.empty or core_file_stats_row['N.cancer.cells'].iloc[0] < 500):
                    continue
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
        tiff_dataset_train = TiffDataset(files=cores_train, channels=channels, labels=labels_train, transform=transform)
        tiff_dataset_test = TiffDataset(files=cores_test, channels=channels, labels=labels_test, transform=transform)

        train_loader = MultiEpochsDataLoader(
            tiff_dataset_train, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=num_workers, collate_fn=None)
        test_loader = MultiEpochsDataLoader(
            tiff_dataset_test, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=num_workers, collate_fn=None)

        # Prepare for training
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # W&B initialization for this fold
        wandb.init(project='pixel_ai', name=f"vae_classifier_fold_{fold_num}", resume="never")

        # Training loop for current fold
        best_f1_test = 0
        best_acc_test = 0
        min_test_loss = float("inf")

        for epoch in range(epochs):
            loss_train, acc_train, f1_train = train_test(
                vae_model, classifier_model, optimizer, train_loader, epoch, train=True, criterion=criterion
            )
            loss_test, acc_test, f1_test = train_test(
                vae_model, classifier_model, optimizer, test_loader, epoch, train=False, criterion=criterion
            )

            # Update best metrics
            best_f1_test = max(best_f1_test, f1_test)
            best_acc_test = max(best_acc_test, acc_test)
            min_test_loss = min(min_test_loss, loss_test)

            # Log metrics
            wandb.log({
                "loss_train": loss_train, "loss_test": loss_test,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "acc_train": acc_train, "acc_test": acc_test,
                "f1_train": f1_train, "f1_test": f1_test
            })

            scheduler.step()
            print(f'Epoch-{epoch} lr: {optimizer.param_groups[0]["lr"]}')
            print(f'Loss test {loss_test}')

        # Store fold-level metrics
        best_fold_accuracies.append(best_acc_test)
        best_fold_f1_scores.append(best_f1_test)
        min_fold_test_losses.append(min_test_loss)

        # Finish W&B run for this fold
        wandb.finish()

    # Compute cross-validation summary
    avg_cv_accuracy = np.mean(best_fold_accuracies)
    avg_cv_f1_score = np.mean(best_fold_f1_scores)
    avg_min_cv_test_loss = np.mean(min_fold_test_losses)

    print("Cross-validation completed.")
    print(f"Average CV Accuracy: {avg_cv_accuracy}")
    print(f"Average CV F1 Score: {avg_cv_f1_score}")
    print(f"Average Minimum CV Test Loss: {avg_min_cv_test_loss}")

    # Log final cross-validation scores to W&B
    wandb.init(project='pixel_ai', name="cv_vae_classifier", resume="never")
    wandb.log({
        "avg_cv_accuracy": avg_cv_accuracy,
        "avg_cv_f1_score": avg_cv_f1_score,
        "avg_min_cv_test_loss": avg_min_cv_test_loss
    })


if __name__ == "__main__":
    main()