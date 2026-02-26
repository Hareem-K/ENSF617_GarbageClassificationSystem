"""
model.py

Purpose:
---------
Defines the multimodal neural network architecture for the garbage
classification task.

This model uses:
- A PRE-TRAINED CNN for image feature extraction
- A (simple or pre-trained) text encoder
- Feature fusion (concatenation)
- A small classifier head trained on the garbage dataset

IMPORTANT (from instructor clarification):
------------------------------------------
- The image encoder SHOULD be pre-trained (e.g. ResNet, EfficientNet, ViT)
- We are STILL training on our Assignment 1 garbage images
- Pre-training only provides a better initialization

This file should ONLY define the model architecture.
NO training loops or data loading logic here.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


class GarbageClassifier(nn.Module):
    """
    Multimodal Garbage Classification Model

    Architecture:
    -------------
    Image (Assignment 1 photo)
        -> Pre-trained CNN (feature extractor)
        -> Image feature vector

    Text (metadata / description)
        -> Text encoder
        -> Text feature vector

    [Image features | Text features]
        -> Fully connected classifier
        -> Class logits
    """

    def __init__(self, num_classes: int, text_feature_dim: int = 128):
        """
        Args:
        -----
        num_classes (int): number of output classes (e.g. 4)
        text_feature_dim (int): dimensionality of text feature vector (depends on dataset / encoding choice)
        """
        super(GarbageClassifier, self).__init__()

        # -------------------------------------------------
        # IMAGE ENCODER (PRE-TRAINED CNN)
        # -------------------------------------------------
        # Using a pre-trained ResNet as feature extractor
        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final classification layer
        image_feature_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        # Optionally freeze CNN weights (can be unfrozen later)
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # TEXT ENCODER
        # -------------------------------------------------
        # NOTE:
        # - This can be a simple embedding + FC layer
        # - OR replaced with a pre-trained text model if desired
        self.text_encoder = nn.Sequential(
            nn.Linear(text_feature_dim, 128),
            nn.ReLU()
        )

        text_embedding_dim = 128

        # -------------------------------------------------
        # FEATURE FUSION + CLASSIFIER
        # -------------------------------------------------
        fused_feature_dim = image_feature_dim + text_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text):
        """
        Forward pass

        Args:
        -----
        image : torch.Tensor
            Image tensor of shape [batch_size, 3, H, W]

        text : torch.Tensor
            Text tensor of shape [batch_size, text_feature_dim]

        Returns:
        --------
        logits : torch.Tensor
            Raw class scores of shape [batch_size, num_classes]
        """

        # Extract image features
        image_features = self.image_encoder(image)

        # Extract text features
        text_features = self.text_encoder(text)

        # Concatenate features
        fused_features = torch.cat((image_features, text_features), dim=1)

        # Classification
        logits = self.classifier(fused_features)

        return logits