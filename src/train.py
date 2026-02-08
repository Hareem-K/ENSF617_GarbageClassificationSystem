"""
train.py

Purpose:
---------
Contains the training and validation loops.

This file is responsible for:
- Training the model on the training set
- Evaluating performance on the validation set
- Tracking loss and accuracy over epochs

IMPORTANT:
----------
- Do NOT define the model here
- Do NOT load raw data here
- This file should be callable from the notebook
"""

import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the model and evaluates on the validation set.

    Returns:
    --------
    history : dict containing:
        - train_loss
        - val_loss
        - train_accuracy
        - val_accuracy
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # TODO:
    # - Loop over epochs
    # - Training step
    # - Validation step
    # - Store metrics in history dict
    # - Print progress per epoch

    return history
