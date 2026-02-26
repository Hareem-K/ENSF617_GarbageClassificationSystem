"""
preprocessing.py

Purpose:
---------
Defines the data augmentation and pre-processing pipelines for the
multimodal garbage classification project.
"""

from torchvision import transforms


def get_image_transforms():
    """
    Returns a dictionary of torchvision transforms for the train, validation,
    and test splits.

    The transforms resize images to 224x224, apply random horizontal flipping
    (for training only) to prevent overfitting, convert images to PyTorch Tensors,
    and normalize them using ImageNet statistics for the pre-trained ResNet.
    """

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    return image_transforms