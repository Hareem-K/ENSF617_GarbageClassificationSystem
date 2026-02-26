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
from tqdm import tqdm

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

    for epoch in range(num_epochs):

        # ---------------------------------------------------------
        # 1. TRAINING PHASE
        # ---------------------------------------------------------
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Wrap the dataloader with tqdm for the progress bar
        train_loop = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)

        for images, texts, labels in train_loop:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Track statistics
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update the progress bar with the current batch loss
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # ---------------------------------------------------------
        # 2. VALIDATION PHASE
        # ---------------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_loop = tqdm(
            val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ", leave=False)

        with torch.no_grad():
            for images, texts, labels in val_loop:
                images = images.to(device)
                texts = texts.to(device)
                labels = labels.to(device)

                outputs = model(images, texts)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # ---------------------------------------------------------
        # 3. STORE AND PRINT METRICS
        # ---------------------------------------------------------
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the final summary for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | "f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    return history