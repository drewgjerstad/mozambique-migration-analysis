"""
This module contains functions for training different classifiers.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.models.eval import compute_metrics_dict

def train_sklearn_model(model, param_grid, X_train, y_train, X_val, y_val):
    """Train Scikit-Learn classifier with class weights for handling the
    imbalanced dataset and with hyperparameter optimization."""

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Print Best Found Hyperparameters
    print("Best Hyperparameters:")
    for param, val in grid_search.best_params_.items():
        print(f"{param}: {val}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    # Evaluate Using Validation Set
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics_dict(y_val, y_val_pred, y_val_prob)

    # Print Validation Set Performance
    print("\nValidation Set Performance:")
    for metric, val in metrics.items():
        print(f"{metric}: {val:.4f}")

    # Print Feature Importance Analysis
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("T\nop 10 Most Important Features")
        print(feature_importance.head(10))

    return best_model, metrics
