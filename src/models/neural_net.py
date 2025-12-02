"""
This module contains functions for a neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

class NeuralNetwork(nn.Module):
    """
    Neural Network with imbalanced data handling.
     * Dropout (default=0.3) prevents overfitting the majority class
     * Batch normalization stabilizes training with imbalanced batches
    """
    def __init__(self, input_dim, hidden_dims=None, num_classes=2, dropout=0.3):
        super(NeuralNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward run through network."""
        return self.network(x)


class CustomDataset(Dataset):
    """Custom Dataset wrapper for migration data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_class_weights(y):
    """Compute balanced class weights for loss function."""
    classes = np.unique(y)
    class_counts = np.bincount(y)
    weights = len(y) / (len(classes) * class_counts)
    return torch.FloatTensor(weights)


def train_neural_net(X_train, y_train, X_val, y_val, n_epochs=50,
                     batch_size=256, learning_rate=0.001):
    """Train neural network with class-weighted loss."""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Create datasets and loaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize neural net
    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.3
    ).to(device)

    # Define loss function, optimizer, learning rate schedule
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation performance
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Compute metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f},"
              f" Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Add early stopping (prevent overfitting of majority class)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, history


def evaluate_neural_net(model, X_test, y_test):
    """Evaluate trained neural network."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Test dataset and loader
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Prediction storage
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'roc_auc': roc_auc_score(all_labels, all_preds)
    }

    return metrics
