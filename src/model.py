"""
model.py

Purpose:
---------
Defines the neural network architecture for the garbage classification model.

The model MUST:
---------------
- Use a CNN for image feature extraction
- Use a separate encoder for text features
- Fuse image + text features before classification
- Output logits for multi-class classification

IMPORTANT:
----------
- Do NOT include training or evaluation loops here
- Do NOT load data here
- Keep the forward pass clean and readable
"""

import torch
import torch.nn as nn

class GarbageClassifier(nn.Module):
    """
    Multimodal garbage classification model.

    Architecture:
    -------------
    Image -> CNN -> image_features
    Text  -> Text Encoder -> text_features
    Concatenate(image_features, text_features)
    -> Fully Connected Classifier
    """

    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()

        # TODO:
        # - Define CNN layers for image branch
        # - Define text encoder (embedding / FC layers)
        # - Define fusion + classifier layers
        pass

    def forward(self, image, text):
        """
        Args:
        -----
        image : torch.Tensor
        text  : torch.Tensor

        Returns:
        --------
        logits : torch.Tensor of shape [batch_size, num_classes]
        """
        pass
