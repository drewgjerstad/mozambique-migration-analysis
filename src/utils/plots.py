"""
This module contains functions for visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_confusion_matrix(y_true, y_pred, labels=None):
    """Generate confusion matrix given true and predicted labels."""
    if not labels:
        labels = np.unique(y_true).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def plot_roc_curve(tpr, fpr, auc=None):
    """Generate ROC curve given true and probabilistic labels."""
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, c='b')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if auc:
        plt.title(f"ROC Curve (AUC: {auc})")
    else:
        plt.title("ROC Curve")
    plt.grid(True)
    plt.show()
