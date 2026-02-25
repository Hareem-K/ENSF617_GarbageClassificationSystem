"""
dataset.py

Purpose:
---------
Defines the PyTorch Dataset for the garbage classification project.

This file:
- Loads images from disk
- Generates simple text features
- Maps class names to integer labels
- Returns (image_tensor, text_tensor, label)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class GarbageDataset(Dataset):
    """
    Custom PyTorch Dataset for multimodal garbage classification.

    Each item returns:
        image_tensor : torch.Tensor
        text_tensor  : torch.Tensor
        label        : int
    """

    def __init__(self, split, transform=None, dataset_root=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: torchvision transforms
            dataset_root (str): root directory of dataset
        """
        assert split in ["train", "val", "test"], "Invalid split name"

        self.split = split
        self.transform = transform

        # Allow flexible dataset path
        self.dataset_root = dataset_root or os.getenv(
            "DATASET_ROOT",
            "/work/TALC/ensf617_2026w/garbage_data"
        )

        self.split_path = os.path.join(self.dataset_root, split)

        # Class names
        self.classes = ["Black", "Blue", "Green", "Other"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all samples
        self.samples = []

        for class_name in self.classes:
            class_folder = os.path.join(self.split_path, class_name)
            if not os.path.exists(class_folder):
                continue

            for filename in os.listdir(class_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepath = os.path.join(class_folder, filename)
                    label = self.class_to_idx[class_name]
                    self.samples.append((filepath, label))

        print(f"{split.upper()} split loaded with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # --- Simple text feature generation ---
        # Since dataset may not include real metadata,
        # we encode class name as one-hot text vector
        text_feature_dim = len(self.classes)
        text_tensor = torch.zeros(text_feature_dim)
        text_tensor[label] = 1.0

        return image, text_tensor, label