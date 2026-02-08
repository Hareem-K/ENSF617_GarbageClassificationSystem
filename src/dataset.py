"""
dataset.py

Purpose:
---------
Defines the PyTorch Dataset and DataLoader logic for the garbage
classification project.

This file is responsible for:
- Loading image files from disk
- Loading or generating associated text data
- Mapping class names to integer labels
- Returning data in a format usable by the model

IMPORTANT:
----------
- Do NOT perform training logic here
- Do NOT define the model here
- This file should ONLY handle data loading & preprocessing
"""

from torch.utils.data import Dataset
from PIL import Image
import os

class GarbageDataset(Dataset):
    """
    Custom PyTorch Dataset for multimodal garbage classification.

    Each item should return:
    ------------------------
    - image_tensor : torch.Tensor
    - text_tensor  : torch.Tensor (or encoded text representation)
    - label        : int
    """

    def __init__(self, split, transform=None):
        """
        Args:
        -----
        split (str): One of ['train', 'val', 'test']
        transform: torchvision transforms applied to images
        """
        self.split = split
        self.transform = transform

        # TODO:
        # - Define dataset root path
        # - Load file paths
        # - Load or define text metadata
        # - Map class names to labels
        pass

    def __len__(self):
        """Return the number of samples in this split."""
        pass

    def __getitem__(self, idx):
        """
        Returns:
        --------
        image : torch.Tensor
        text  : torch.Tensor
        label : int
        """
        pass
