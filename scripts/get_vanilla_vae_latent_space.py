import torch; torch.manual_seed(0)
import torch.distributions
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pandas as pd
import ipdb
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
epochs = 30
batch_size = 4096
lr = 0.0001
latent_dims = 32
hidden_dims = [32, 64, 128, 256, 512]
best_loss_val = 99999999
transform_to_image = T.ToPILImage()
cores_folder = 'TMA_18_810/'
files_path='/data/projects/pixel_project/datasets/NKI_project_TMAs/patches/randomly_generated/{0}'.format(cores_folder)
model_path='/data/projects/pixel_project/saved_models'
model_name = "allcores_randompatches_vae"
input_dimensions = (32, 32)

model = VanillaVAE(in_channels=38,latent_dim=latent_dims,input_dimensions=input_dimensions,hidden_dims=hidden_dims).to(device) # GPU
checkpoint = torch.load('{}/.pth.tar'.format(model_path))
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
tiff_dataset = TiffDataset(files=patches_files,transform=T.Resize(input_dimensions))

data_loader = torch.utils.data.DataLoader(
    tiff_dataset, batch_size=batch_size, 
    shuffle=None, pin_memory=True, 
    sampler=None
    )

# Try to generate Latent space :/
for i, images in enumerate(data_loader):
    images = images.cuda()
    x_hat = model(images)
    loss = model.loss_function(*x_hat, M_N=0.00025)
    z = x_hat[4]
    value = z.cpu().detach().numpy()

np.save('../data/{}latent_space.npy'.format(model_name), value)