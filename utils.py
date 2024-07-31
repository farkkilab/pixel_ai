import tifffile
import numpy as np
import os
import ipdb
import shutil
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import cv2
import numpy as np
from typing import Optional, Union

ERROR_THRESHOLD = "Threshold should be in range [0, 255], got {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2

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


def pad_image(image, target_size=(224, 224)):
    """
    Pads the image to the target size with zeros.

    Args:
        image (torch.Tensor): The input image tensor.
        target_size (tuple): The desired output size (height, width).

    Returns:
        torch.Tensor: The padded image tensor.
    """
    channels, height, width = image.shape[-3],image.shape[-2], image.shape[-1]
    target_h, target_w = target_size

    if height < target_h or width < target_w:
        padded_image = np.zeros((channels, target_h, target_w), dtype=image.dtype)
        padded_image[:,:height, :width] = image
    else:
        padded_image = image

    return padded_image

class TiffDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, files_names=None,transform=None, channels=None, labels=None):
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
        self.channels = channels
        self.labels = labels

    def __len__(self):
        return len(self.tiff_files)
    def get_file_name(self, idx):
        return self.files_names[idx]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_tiff = tifffile.imread(self.tiff_files[idx],key=self.channels,maxworkers=32)
        #info = np.iinfo(im_tiff.dtype)
        sample = torch.from_numpy(im_tiff / 65535).float()
        if self.transform:
            sample = self.transform(sample)

        #sample = torch.from_numpy(np.log1p(im_tiff)).float()
        #array_expression = np.array([array_expression])
        #array_expression = array_expression.astype('float32').reshape(-1, 1657)
        #array_expression = np.pad(array_expression, (0, 7), 'constant')
        #array_expression = array_expression[:,  None, :, :]
        #sample = torch.from_numpy(im_tiff)



        output = []
        output.append(sample)
        if self.files_names:
            output.append(self.files_names[idx])
        if self.labels:
            output.append(self.labels[idx])
        return tuple(output)

class TensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, files_names=None,transform=None, channels=None, labels=None, gigapath=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tensor_files = files
        self.files_names = files_names
        self.transform = transform
        self.labels = labels
        self.gigapath = gigapath

    def __len__(self):
        return len(self.tensor_files)
    def get_file_name(self, idx):
        return self.files_names[idx]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.gigapath:
            embed_dict = torch.load(self.tensor_files[idx])

            # Collect all embeddings and stack them into a tensor of shape (14, 768)
            embeddings = [embed_dict[key] for key in embed_dict.keys()]
            tensor = torch.stack(embeddings)
        else:
            tensor = torch.load(self.tensor_files[idx])
        if self.transform:
            tensor = self.transform(tensor)


        output = []

        output.append(tensor)
        if self.files_names:
            output.append(self.files_names[idx])
        if self.labels:
            output.append(self.labels[idx])
        return tuple(output)

class TensorDataset2D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, slides, files_names=None,transform=None, channels=None, labels=None, gigapath=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.slides = slides
        self.files_names = files_names
        self.transform = transform
        self.labels = labels
        self.gigapath = gigapath
    def __len__(self):
        return len(self.slides)
    def get_file_name(self, idx):
        return self.files_names[idx]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patches_tensors = [(file, (int(file.split('_')[1]),int(file.split('_')[2]))) for i,file in enumerate(os.listdir(self.slides[idx])) if file.endswith('_tensor.pt')]
        patches_tensors.sort(key=lambda item: (item[1][1], item[1][0]))  # Sort by Y first, then by X
        M = 8
        N = 8
        if self.gigapath:
            embedding_dim = 1536
        else:
            embedding_dim = 1024

        # Create an empty tensor to store the embeddings
        embeddings = torch.zeros((M, N, embedding_dim))

        for i in range(M):
            for j in range(N):
                if i+j <len(patches_tensors):
                    patch = patches_tensors[i+j]
                    embeddings[i, j] = torch.load(os.path.join(self.slides[idx],patch[0]))

        tensor = embeddings
        if self.transform:
            tensor = self.transform(tensor)


        output = []

        output.append(tensor)
        if self.files_names:
            output.append(self.files_names[idx])
        if self.labels:
            output.append(self.labels[idx])
        return tuple(output)


class TensorDatasetMIL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, slides, files_names=None,transform=None, labels=None, raw_images=None,gigapath=None, channels=None, multi_channels=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.slides = slides
        self.files_names = files_names
        self.transform = transform
        self.labels = labels
        self.raw_images = raw_images
        self.gigapath = gigapath
        self.channels = channels
        self.multi_channels = multi_channels
    def __len__(self):
        return len(self.slides)
    def get_file_name(self, idx):
        return self.files_names[idx]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patches_tensors = []
        if self.multi_channels:
            # if there are two mentions to the same slide and core, combine them under same bag (will be different channel image for same sample)
            slides_dirs = []
            for channels in os.listdir('/'.join(self.slides[idx].split('/')[:-3])):
                slides_dirs.append(os.path.join(os.path.join('/'.join(self.slides[idx].split('/')[:-3]), channels),'TMA'+self.slides[idx].split('TMA')[-1]))
        else:
            slides_dirs = [self.slides[idx]]
        for slide in slides_dirs:
            patches_tensors.extend([(os.path.join(slide,file), (int(file.split('_')[1]),int(file.split('_')[2]))) for i,file in enumerate(os.listdir(slide)) if file.endswith('_tensor.pt')])
        patches_tensors.sort(key=lambda item: (item[1][1], item[1][0]))  # Sort by Y first, then by X
        patches_files = []
        for file in patches_tensors:
            patches_files.append(file[0])
        if self.raw_images:
            patches_images = [(file, (int(file.split('_')[1]), int(file.replace('.tiff','').split('_')[2]))) for i, file in
                               enumerate(os.listdir(self.raw_images[idx])) if file.endswith('.tiff')]
            patches_images.sort(key=lambda item: (item[1][1], item[1][0]))  # Sort by Y first, then by X
            patches_images_files = []
            for file in patches_images:
                patches_images_files.append(os.path.join(self.raw_images[idx], file[0]))

        if self.gigapath:
            embedding_dim = 1536
        else:
            embedding_dim = 1024

        data = []
        output = []
        for i in range(len(patches_tensors)):
            data.append(torch.load(patches_tensors[i][0]))
        tensor = torch.stack(data)

        output.append(tensor)
        if self.raw_images:
            raw_images_data = []
            for i in range(len(patches_images_files)):
                #raw_images_data.append(torch.tensor(tifffile.imread(patches_images_files[i],key=self.channels,maxworkers=32)).float())
                image = tifffile.imread(patches_images_files[i],key=self.channels,maxworkers=28)
                #normalized_image = image / 65535
                # Apply log transformation (add 1 to avoid log(0))
                #log_transformed_image = torch.from_numpy(np.log1p(normalized_image)).float()
                padded_image = pad_image(image)
                padded_image = torch.from_numpy(padded_image).float()

                raw_images_data.append(padded_image)
            raw_images_data_tensor = torch.stack(raw_images_data)
            if self.transform:
                raw_images_data_tensor = self.transform(raw_images_data_tensor)
            output.append(raw_images_data_tensor)
        else:
            if self.transform:
                tensor = self.transform(tensor)





        if self.files_names:
            output.append(self.files_names[idx])
        if self.labels:
            output.append(torch.as_tensor(self.labels[idx], dtype=torch.float32))
        output.append(patches_files)
        return tuple(output)


def collate_fn_MIL(batch):
    bags, labels = zip(*batch)
    max_length = max(len(bag) for bag in bags)
    padded_bags = []
    mask = []

    for bag in bags:
        padded_bag = torch.cat([bag, torch.zeros(max_length - len(bag), bag.size(1))], dim=0)
        padded_bags.append(padded_bag)
        mask.append(torch.cat([torch.ones(len(bag)), torch.zeros(max_length - len(bag))], dim=0))

    padded_bags = torch.stack(padded_bags)
    mask = torch.stack(mask)
    labels = torch.stack(labels)

    return padded_bags, labels, mask

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


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

def create_random_patches_wholeslide(z, patch_size, output_folder, top_left_y, top_left_x):
    # Get the dimensions of the image

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract the patch from the image
    patch = z[0][:, top_left_y:top_left_y + patch_size[0], top_left_x:top_left_x + patch_size[1]]
    # Save the patch as a new TIFF file
    patch_filename = f"{output_folder}/patch_{top_left_y}_{top_left_x}.tiff"
    tifffile.imwrite(patch_filename, patch)


def get_patch_stats(patch_path):
    with tifffile.TiffFile(patch_path) as tif:
        patch_image = tif.asarray()

    #patch_image = tifffile.imread(patch_path)
    patch_file = patch_path.split('/')[-1].lower()
    patch_core = patch_path.split('/')[-2].lower()
    patch_slide = patch_path.split('/')[-3]
    median = np.median(patch_image[0, :, :])
    mean = np.mean(patch_image[0, :, :])
    standard_deviation = np.std(patch_image[0, :, :])
    return median, mean, standard_deviation, patch_file, patch_core, patch_slide

def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma <= SIGMA_NO_OP:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def _otsu_threshold(*, gray: np.ndarray) -> int:
    """Helper function to calculate Otsu's thresold from a grayscale image."""
    values = gray.flatten()
    values = values[(values != WHITE_PIXEL) & (values != BLACK_PIXEL)]
    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold
def get_tissue_mask(
    image: Union[Image.Image, np.ndarray],
    *,
    threshold: Optional[int] = None,
    multiplier: float = 1.0,
    sigma: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Detect tissue from image.

    Args:
        image: Input image.
        threshold: Threshold for tissue detection. If set, will detect tissue by
            global thresholding, and otherwise Otsu's method is used to find
            a threshold. Defaults to None.
        multiplier: Otsu's method is used to find an optimal threshold by
            minimizing the weighted within-class variance. This threshold is
            then multiplied with `multiplier`. Ignored if `threshold` is not None.
            Defaults to 1.0.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.

    Raises:
        ValueError: Threshold not between 0 and 255.

    Returns:
        Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check arguments.
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    # Convert to grayscale.
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blurring.
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold is None:
        threshold = _otsu_threshold(gray=blur)
        threshold = max(min(255, int(threshold * max(0.0, multiplier) + 0.5)), 0)
    # Global thresholding.
    thrsh, mask = cv2.threshold(blur, threshold, 1, cv2.THRESH_BINARY)
    return int(thrsh), mask


def load_tensors_from_directory(tensors_path):
    # List all .pt files in the directory
    tensor_files = [os.path.join(tensors_path,f)  for f in os.listdir(tensors_path) if f.endswith('.pt')]
    tensors = [torch.load(f) for f in tensor_files]
    multi_dim_tensor = torch.stack(tensors, dim=0)
    coords = [torch.tensor([int(file.split('patch_')[1].split('_')[0]),int(file.split('patch_')[1].split('_')[1])]) for file in tensor_files]
    coords_tensor = torch.stack(coords, dim=0)
    return multi_dim_tensor, coords_tensor


def save_tile(xywh, im_tiff, output_dir):
    x, y, w, h = xywh
    patch = im_tiff[:, y:y + h, x:x + w]
    # Save the patch as a new TIFF file
    patch_filename = f"{output_dir}/patch_{y}_{x}.tiff"
    tifffile.imwrite(patch_filename, patch)


def remove_part_by_negative_index(path, negative_index):
    parts = path.split('/')
    if abs(negative_index) <= len(parts):
        parts.pop(negative_index)
    return '/'.join(parts)

def get_percentiles_normalize(directories_path, channels, min_percentil=1, max_percentil=99):
    images = []
    for dir_path in directories_path:
        for file_path in os.listdir(dir_path):
            if file_path.endswith('tiff') or file_path.endswith('tif'):
                img = tifffile.imread(os.path.join(dir_path,file_path), key=channels)
                padded_image = pad_image(img, (224, 224))
                images.append(padded_image)

    images = np.stack(images)  # Shape will be (500, 224, 224, channel_number)
    reshaped_images = images.reshape(images.shape[0], images.shape[1], -1)
    percentile_1 = np.percentile(reshaped_images, min_percentil, axis=(0, 2))
    percentile_99 = np.percentile(reshaped_images, max_percentil, axis=(0, 2))
    return percentile_1, percentile_99
class PercentileNormalize(object):
    def __init__(self, percentile_min, percentile_max):
        self.percentile_min = torch.tensor(percentile_min, dtype=torch.float32)
        self.percentile_max = torch.tensor(percentile_max, dtype=torch.float32)

    def __call__(self, img):
        # Assuming img is a PyTorch tensor with shape (C, H, W)

        img = (img - self.percentile_min[:, None, None]) / (self.percentile_max[:, None, None] - self.percentile_min[:, None, None])
        img = torch.clamp(img, 0, 1)  # Ensure values are within [0, 1]
        return img

