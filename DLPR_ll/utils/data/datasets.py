import torch
from torch.utils import data

import os
import numpy as np

from PIL import Image


class ImageDataset(data.Dataset):

    def __init__(self, path_dir, img_mode = None, transform=None):
        self.path_dir = path_dir
        self.img_mode = img_mode
        self.transform = transform
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        img_path = os.path.join(self.path_dir, image_name)
        img = Image.open(img_path)

        if self.img_mode is not None:
            img = img.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)

        return img



        crop_tensor = torch.from_numpy(crop_3channel)

        return crop_tensor

import torch
from astropy.io import fits
import numpy as np
import os
import random

class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, fits_folder, crop_size, num_datapoints_per_epoch):
        self.fits_folder = fits_folder
        self.crop_size = crop_size
        self.num_datapoints_per_epoch = num_datapoints_per_epoch
        self.fits_files = [f for f in os.listdir(fits_folder) if f.endswith('.fits')]

    def __len__(self):
        return self.num_datapoints_per_epoch

    def __getitem__(self, idx):
        # Randomly sample a FITS file
        fits_file = random.choice(self.fits_files)
        fits_path = os.path.join(self.fits_folder, fits_file)

        # Load the FITS file
        with fits.open(fits_path) as hdul:
            data = hdul['SCI'].data.astype(np.uint16)  # Ensure uint16 format

        # Randomly sample a crop
        height, width = data.shape
        x = random.randint(0, width - self.crop_size)
        y = random.randint(0, height - self.crop_size)
        crop = data[y:y + self.crop_size, x:x + self.crop_size]

        # Extract upper and lower 8 bits
        crop_upper = (crop >> 8).astype(np.uint8)  # Upper 8 bits
        crop_lower = (crop & 0xFF).astype(np.uint8)  # Lower 8 bits
        crop_zero = np.zeros_like(crop_upper, dtype=np.uint8)  # All 0

        # Stack as a 3-channel image
        crop_3channel = np.stack((crop_upper, crop_lower, crop_zero), axis=0)

        # Convert to PyTorch tensor
        crop_tensor = torch.from_numpy(crop_3channel)

        return crop_tensor.float()