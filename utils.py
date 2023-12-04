import tifffile
import numpy as np
import os
import ipdb
import shutil
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, 'saved_models/'+filename)
    if is_best:
        shutil.copyfile('saved_models/'+filename,
                        'saved_models/model_best_{0}'.format(filename))

class TiffDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, files_names=None,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tiff_files = files
        self.files_names = files_names
        self.transform = transform

    def __len__(self):
        return len(self.tiff_files)
    def get_file_name(self, idx):
        return self.files_names[idx]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_tiff = tifffile.imread(self.tiff_files[idx],
                                  maxworkers=6)
        info = np.iinfo(im_tiff.dtype)
        sample = torch.from_numpy(im_tiff/info.max).float()
        #array_expression = np.array([array_expression])
        #array_expression = array_expression.astype('float32').reshape(-1, 1657)
        #array_expression = np.pad(array_expression, (0, 7), 'constant')
        #array_expression = array_expression[:,  None, :, :]
        #sample = torch.from_numpy(im_tiff)
        if self.transform:
            sample = self.transform(sample)

        if self.files_names:
            return sample, self.files_names[idx]
        else:
            return sample

def create_random_patches(image_path, patch_size, output_folder):
    # Load the TIFF image
    image = tifffile.imread(image_path)

    # Get the dimensions of the image
    num_channels, image_height, image_width = image.shape
    # Remove extra channels
    channel_selected = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 21, 22, 23, 25, 26, 27, 29, 30, 31,
                        33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47]

    if num_channels == 48:
        image = image[channel_selected,:,:]
    # Ensure the patch size is valid
    if patch_size[0] > image_height or patch_size[1] > image_width:
        raise ValueError("Patch size is larger than image dimensions")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all possible patch positions
    for top_left_y in range(0, image_height - patch_size[0] + 1, patch_size[0]):
        for top_left_x in range(0, image_width - patch_size[1] + 1, patch_size[1]):
            # Extract the patch from the image
            patch = image[:, top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]]
            # Save the patch as a new TIFF file
            patch_filename = f"{output_folder}/patch_{top_left_y}_{top_left_x}.tiff"
            tifffile.imwrite(patch_filename, patch)
