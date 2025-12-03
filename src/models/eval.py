"""
This module contains functions for evaluating classifiers.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

def compute_metrics_dict(y_true, y_pred, y_prob) -> dict:
    """
    Compute the accuracy, precision, recall, f1 score, and roc/auc score for a
    classifier given sets of true and predicted response values.
    """

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'roc_curve': roc_curve(y_true, y_prob)
    }

    return metrics


def evaluate_sklearn_model(model, X_test, y_test):
    """Evaluate trained scikit-learn model on test set."""

    # Compute Predictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Compute Metrics
    metrics = compute_metrics_dict(y_test, y_test_pred, y_test_prob)

    return metrics
