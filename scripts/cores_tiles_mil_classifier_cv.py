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
import itertools
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
            bag_tensor, raw_images, labels, files_names = item
            raw_images = raw_images.to(device)
            input_variable = raw_images
        else:
            bag_tensor, raw_images, labels, files_names = item
            bag_tensor = bag_tensor.to(device)
            input_variable = bag_tensor
        labels = labels.to(device)
        data_time.update(time.time() - end)
        #predictions, attention_weights = classifier_model(bag_tensor, mask)
        #predictions, attention_weights, A = classifier_model(bag_tensor)
        if not train:
            with torch.no_grad():
                loss, attention_weights = classifier_model.calculate_objective(input_variable, labels, eval_mode=True)
                error, binary_predictions = classifier_model.calculate_classification_error(input_variable, labels,eval_mode=True)
        else:
            loss, attention_weights = classifier_model.calculate_objective(input_variable, labels)
            error, binary_predictions = classifier_model.calculate_classification_error(input_variable, labels)
        binary_predictions = binary_predictions.cpu().tolist()
        # Compute loss


        if train:
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), 1)
        binary_predictions_list.extend(binary_predictions)
        labels_predictions.extend(labels.cpu())
        batch_time.update(time.time() - end)
        #if i % 100 == 0:
        progress.display(i)
    accuracy_value = accuracy_score(labels_predictions, binary_predictions_list)
    f1_value = f1_score(labels_predictions, binary_predictions_list, average='macro')
    return losses.avg, accuracy_value, f1_value


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device("cuda")
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=Path,
                        default="/scratch/project_2003009/NKI_histoprep_patches/224")
    # cores or whole_slide
    parser.add_argument("--data_type", type=str,
                        default="cores")
    parser.add_argument("--classification_task", type=str,
                        default="nact")
    parser.add_argument("--architecture", type=str,
                        default="amlab_attention")
    parser.add_argument("--filter_cores_lowcancer", type=str,
                        default=False)
    parser.add_argument("--image_normalization", type=str,
                        default=False)
    #use more than 3 channels, look for other channel images of same core and slide
    parser.add_argument("--multi_channels", type=bool,
                        default=False)
    parser.add_argument("--model_encoder", type=str,
                        default="trainable")

    p = parser.parse_args()
    files_path = p.files_path
    data_type = p.data_type
    classification_task = p.classification_task
    architecture = p.architecture
    filter_cores_lowcancer = p.filter_cores_lowcancer
    if filter_cores_lowcancer == 'true':
        filter_cores_lowcancer = True
    else:
        filter_cores_lowcancer = False

    image_normalization = p.image_normalization
    if image_normalization == 'true':
        image_normalization = True
    else:
        image_normalization = False
    multi_channels = p.multi_channels
    model_encoder = p.model_encoder




    best_f1_test = 0
    epochs = 30
    # with image normalizaiton we need a lower lr
    if image_normalization:
        lr = 0.00001
    else:
        lr = 0.00001
    num_workers = 28
    channels = [0, 25, 28]
    model_path = 'saved_models'
    if model_encoder == 'trainable':
        input_dimensions = 224
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
    limited_combinations = slide_combinations[:2]
    best_fold_accuracies = []
    best_fold_f1_scores = []
    min_fold_test_losses = []

    for fold_num, test_slides in enumerate(limited_combinations):
        # Define training slides as those not in the current test pair
        train_slides = [slide for slide in slides_directories if slide not in test_slides]

        # Initialize lists for core directories for this fold
        cores_train_directory = []
        cores_test_directory = []

        # Loop over slides to collect core directories based on current fold's split
        for slide in slides_directories:
            slide_files_path = os.path.join(files_path, slide)
            for r, ds, fs in os.walk(slide_files_path):
                for dn in ds:
                    if dn.startswith('core'):
                        core_path = os.path.join(r, dn)
                        cores_directories.append(core_path)
                        # Assign core to train or test based on whether slide is in test_slides
                        if slide in test_slides:
                            cores_test_directory.append(core_path)
                        else:
                            cores_train_directory.append(core_path)
        cores_chemo_labels_df = pd.read_csv('data/cores_labels_chemotherapy.csv')
        if filter_cores_lowcancer:
            cores_stats_df = pd.read_csv('data/cores_stats_ncancer_cells.csv')
        for i, core_directory in enumerate(cores_directories):
            patch_file_label_df = cores_chemo_labels_df[(cores_chemo_labels_df['cycif.slide']==core_directory.split('/')[-2])&(cores_chemo_labels_df['cycif.core.id']==core_directory.split('/')[-1])]
            if filter_cores_lowcancer:
                core_file_stats_row = cores_stats_df[(cores_stats_df['cycif.slide'] == core_directory.split('/')[-2]) & (
                        cores_stats_df['cycif.core.id'] == core_directory.split('/')[-1])]

            # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
            # Ignore labels with PDS followed by NACT
            if not patch_file_label_df.empty and str(patch_file_label_df.iloc[0]['therapy_sequence']).lower()!='na' and not pd.isnull(patch_file_label_df.iloc[0]['therapy_sequence'])\
                    and not patch_file_label_df.iloc[0]['therapy_sequence'] == 'PDS followed by NACT':
                # skip the rows if filter_cores_lowcancer is set to true and the core don't have enough cancer cells
                if filter_cores_lowcancer and (core_file_stats_row.empty or core_file_stats_row['N.cancer.cells'].iloc[0] < 500):
                    continue
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
                print('Missing label for:'+core_directory)
        config['total_cores'] = len(cores_directories)
        config['train_images'] = len(cores_train)
        config['test_images'] = len(cores_test)
        # , mode="disabled"
        wandb.init(project='pixel_ai', name="embedding_{0}_classifier_mil_cv_tilescore_{1}_encoder_{2}_filterlowcancercells_{3}_imagenormalization_{4}".format(classification_task,str(files_path).split('/')[-1],model_encoder, filter_cores_lowcancer, image_normalization), resume="never", config=config)
        if model_encoder == 'trainable':
            raw_cores_train = [core.replace('histoprep_embeddings_uni','NKI_histoprep_patches').replace('/'+'_'.join(str(channel) for channel in channels),'').replace('/0_0_0_normalization_False','') for core in cores_train]
            raw_cores_test = [core.replace('histoprep_embeddings_uni', 'NKI_histoprep_patches').replace('/'+'_'.join(str(channel) for channel in channels), '').replace('/0_0_0_normalization_False','') for core
                               in cores_test]
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
        tensor_dataset_train = TensorDatasetMIL(slides=cores_train,files_names=cores_train,raw_images=raw_cores_train,transform=transforms_train,labels=labels_train, channels=channels, gigapath=False,multi_channels=multi_channels, image_normalization=image_normalization)
        tensor_dataset_test = TensorDatasetMIL(slides=cores_test,files_names=cores_test,raw_images=raw_cores_test,transform=transforms_test, labels=labels_test, channels=channels, gigapath=False,multi_channels=multi_channels, image_normalization=image_normalization)
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
        best_f1_test = 0  # Track best F1 for this fold
        best_acc_test = 0  # Track best accuracy for this fold
        min_test_loss = float("inf")  # Initialize minimum test loss for this fold

        for epoch in range(epochs):
            loss_train, acc_train, f1_train = train_test(classifier_model,optimizer, train_loader,epoch,train=True, criterion=criterion, model_encoder=model_encoder)
            loss_test, acc_test, f1_test = train_test(classifier_model,optimizer, test_loader, epoch, train=False, criterion=criterion, model_encoder=model_encoder)
            if f1_test > best_f1_test:
                best_f1_test = f1_test
            if acc_test > best_acc_test:
                best_acc_test = acc_test
            if loss_test < min_test_loss:
                min_test_loss = loss_test
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
            }, is_best, 'embedding_{0}_mil_fullcore_{1}_encoder_{2}_filterlowcancercells_{3}_imagenormalization_{4}.pth.tar'.format(classification_task,str(files_path).split('/')[-1],model_encoder,filter_cores_lowcancer, image_normalization))
        best_fold_accuracies.append(best_acc_test)
        best_fold_f1_scores.append(best_f1_test)
        min_fold_test_losses.append(min_test_loss)
        wandb.finish()
    avg_cv_accuracy = np.mean(best_fold_accuracies)
    avg_cv_f1_score = np.mean(best_fold_f1_scores)
    avg_min_cv_test_loss = np.mean(min_fold_test_losses)

    print(f"Cross-validation completed.")
    print(f"Average CV Accuracy: {avg_cv_accuracy}")
    print(f"Average CV F1 Score: {avg_cv_f1_score}")
    print(f"Average Minimum CV Test Loss: {avg_min_cv_test_loss}")

    # Log final cross-validation scores to W&B
    wandb.log({"avg_cv_accuracy": avg_cv_accuracy, "avg_cv_f1_score": avg_cv_f1_score,
               "avg_min_cv_test_loss": avg_min_cv_test_loss})

if __name__ == "__main__":
    main()
