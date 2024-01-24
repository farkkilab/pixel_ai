import torch; torch.manual_seed(0)
import torch.distributions
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pandas as pd
import ipdb
import random
import os, sys, glob
from sklearn.model_selection import train_test_split
import seaborn as sns
import json
from sklearn.manifold import TSNE
import pathlib
from collections import Counter
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from models.vanilla_vae import VanillaVAE
from utils import ProgressMeter, AverageMeter, save_checkpoint, TiffDataset


# declaring the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 250
batch_size = 256
lr = 0.001
latent_dims = 8
channels = [0, 1, 2, 25, 27, 29]
in_channels = len(channels)
hidden_dims = [ 16, 32, 64, 128, 256]
best_loss_val = 99999999
kld_weight = 0.000025
input_dimensions = (128, 128)
transform_to_image = T.ToPILImage()
cores_folder = 'TMA_18_810/'
files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
model_path='/data/projects/pixel_project/saved_models'
model_name = "model_best_allcores_randompatches_[0, 1, 2, 25, 27, 29]"

model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dims, input_dimensions=input_dimensions, hidden_dims=hidden_dims).to(device) # GPU
checkpoint = torch.load('{}/{}_vae.pth.tar'.format(model_path, model_name))

for key in list(checkpoint['state_dict'].keys()):
    if 'module.' in key:
        checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
        del checkpoint['state_dict'][key]

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# load the patches
patches_files = [os.path.join(r, fn)
        for r, ds, fs in os.walk(files_path)
        for fn in fs if fn.endswith('.tiff')]
tiff_dataset = TiffDataset(files=patches_files, 
transform=T.Resize(input_dimensions), 
channels=channels) # random.choices(patches_files, k=4000)

data_loader = torch.utils.data.DataLoader(
    tiff_dataset, batch_size=batch_size, 
    shuffle=None, pin_memory=True, 
    sampler=None
    )

# Calculate and save Latent space
latent_list = []

for i, images in enumerate(data_loader):
    images = images.cuda()
    x_hat = model(images)
    loss = model.loss_function(*x_hat, M_N=kld_weight)
    z = x_hat[4]
    value = z.cpu().detach().numpy()
    latent_list.append(value)

all_value = np.vstack(latent_list)
np.save('../data/{}_latent_space.npy'.format(model_name), all_value)