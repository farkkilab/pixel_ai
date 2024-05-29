from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, transforms, models
import torchvision.transforms as T
import torch
import torch.nn as nn
import ipdb
import ssl
import os
import argparse
from pathlib import Path
import cv2
import pandas as pd
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset, MultiEpochsDataLoader


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
ssl._create_default_https_context = ssl._create_unverified_context
classifier_model = models.resnet50(pretrained=True)
num_features = classifier_model.fc.in_features
classifier_model.fc = nn.Linear(num_features, 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('saved_models/model_best_vit_chemo_classifier_fullcore.pth.tar')
start_epoch = checkpoint['epoch']
classifier_model.load_state_dict(checkpoint['state_dict'])

# Move model to device
classifier_model = classifier_model#.to(device)
target_layers = [classifier_model.layer4[-1]]

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

labels_train = []
labels_test = []
labels_validate = []
# Only files that we have a label for
files_train = []
files_test = []
if data_type == 'cores':
    for i, core_file in enumerate(cores_files):
        patch_file_label_df = cores_chemo_labels_df[
            (cores_chemo_labels_df['cycif.slide'] == core_file.split('/')[-3]) & (
                        cores_chemo_labels_df['cycif.core.id'] == core_file.split('/')[-1].replace('.tif', ''))]
        core_file_stats_row = cores_stats_df[(cores_stats_df['cycif.slide'] == core_file.split('/')[-3]) & (
                cores_stats_df['cycif.core.id'] == core_file.split('/')[-1].replace('.tif', ''))]
        # if core_file_stats is empty, we assume that there is no cancer cells in the core and we should skip it
        if not patch_file_label_df.empty and not core_file_stats_row.empty and str(
                patch_file_label_df.iloc[0]['therapy_sequence']).lower() != 'na' and not pd.isnull(
                patch_file_label_df.iloc[0]['therapy_sequence']):

            if core_file_stats_row['N.cancer.cells'].iloc[0] > 500:
                if core_file.split('/')[-3] == 'TMA_42_961':
                    files_test.append(core_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                else:  # if core_file.split('/')[-3]=='TMA_44_810' or core_file.split('/')[-3]=='TMA_45_312':
                    files_train.append(core_file)
                    # If contains NACT, is a sample collected after chemotherapy exposure
                    if 'nact' in str(patch_file_label_df.iloc[0]['therapy_sequence']).lower():
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
        else:
            print('Missing label for:' + core_file)
transform_to_image = T.ToPILImage()
input_dimensions = (1024, 1024)
batch_size = 1
num_workers = 0
#checkpoint = torch.load('{}/{}_vae.pth.tar'.format(model_path, model_name))
transforms_train = torch.nn.Sequential(
    T.CenterCrop(2048),
    T.RandomCrop(input_dimensions[0])
    #T.Resize([input_dimensions[0], input_dimensions[1]])
)
transforms_test = torch.nn.Sequential(
    T.CenterCrop(2048),
    # T.CenterCrop(input_dimensions[0]),
    T.Resize([input_dimensions[0], input_dimensions[1]])
)
channels = [0, 12, 28]
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
# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:

cam = GradCAM(model=classifier_model, target_layers=target_layers)

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = None#[ClassifierOutputTarget(281)]
for i, (images, labels) in enumerate(test_loader):
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.

    targets_cam = [ClassifierOutputTarget(tar) for tar in labels.tolist()]
    grayscale_cam = cam(input_tensor=images, targets=targets_cam)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    image_np = images[0].cpu().numpy().transpose(1, 2, 0)
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    cv2.imwrite('activations_roi/{0}_gradcam_classifier.jpg'.format(str(i)), visualization)
    cv2.imwrite('activations_roi/{0}_base_classifier.jpg'.format(str(i)), image_np)
    #cv2.imshow('visualization', visualization)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # You can also get the model outputs without having to re-inference
    model_outputs = cam.outputs
