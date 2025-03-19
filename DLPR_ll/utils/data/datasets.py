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
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob

class AstroDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing .npy files.
            transform (callable, optional): Optional transform to be applied to each sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.filepaths = glob(os.path.join(data_dir, "*.npy"))  # List all .npy files

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        npy_path = self.filepaths[idx]
        image = np.load(npy_path)  # Load the .npy file as a numpy array

        # Convert to a PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([0])
