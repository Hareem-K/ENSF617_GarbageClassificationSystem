"""
evaluate.py

Purpose:
---------
Handles model evaluation and visualization.

This file is responsible for:
- Evaluating the model on the test set
- Returning predictions and true labels
- Visualizing incorrect classifications

IMPORTANT:
----------
- Do NOT train the model here
- This file is used AFTER training
"""

import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    """
    Runs inference on the test set.

    Returns:
    --------
    y_true : list of true labels
    y_pred : list of predicted labels
    """
    # TODO:
    # - Disable gradients
    # - Loop through test_loader
    # - Collect predictions and labels
    pass


def show_incorrect_predictions(model, dataset, classes, device, num_examples=8):
    """
    Displays examples where the model prediction != true label.

    Args:
    -----
    model       : trained model
    dataset     : test dataset
    classes     : list of class names
    num_examples: number of incorrect samples to display
    """
    # TODO:
    # - Identify incorrect predictions
    # - Plot images with true vs predicted labels
    pass
