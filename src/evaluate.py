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
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)

            # Apply Softmax to get probabilities
            probs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            # Convert to CPU and python lists/arrays
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())  # Save probabilities

    # Return y_probs as a numpy array
    return y_true, y_pred, np.array(y_probs)


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
    model.eval()
    incorrect_images = []
    incorrect_preds = []
    true_labels = []

    with torch.no_grad():
        # Iterate through the dataset one by one to find mismatches
        for i in range(len(dataset)):
            image, text, label = dataset[i]

            # Add batch dimension and move to device
            img_in = image.unsqueeze(0).to(device)
            txt_in = text.unsqueeze(0).to(device)

            output = model(img_in, txt_in)
            pred = output.argmax(dim=1).item()

            if pred != label:
                # Un-normalize the image for matplotlib visualization
                img_plot = image.permute(1, 2, 0).numpy()

                # Reverse the ImageNet normalization applied in preprocessing
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_plot = std * img_plot + mean
                img_plot = np.clip(img_plot, 0, 1)

                incorrect_images.append(img_plot)
                incorrect_preds.append(classes[pred])
                true_labels.append(classes[label])

            if len(incorrect_images) >= num_examples:
                break

    # Plot the incorrect examples in a grid
    cols = 4
    rows = (num_examples + cols - 1) // cols
    plt.figure(figsize=(15, 4 * rows))

    for i in range(len(incorrect_images)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(incorrect_images[i])
        # Display the actual and predicted labels in red
        plt.title(
            f"Actual: {true_labels[i]}\nPredicted: {incorrect_preds[i]}", color='red', fontweight='bold')
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_multiclass_roc_curve(y_true, y_probs, classes):
    """
    Plots the One-vs-Rest ROC curve for a multi-class classification problem.
    """
    # Binarize the true labels (e.g., class 2 becomes [0, 0, 1, 0])
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    # Matched roughly to your classes
    colors = cycle(['black', 'blue', 'green', 'orange'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve: {classes[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot the random guess diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Multi-Class ROC Curve (One-vs-Rest)', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()