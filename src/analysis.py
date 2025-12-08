"""
This script is used to generate analyses for our models on MSI GPUs.
"""

from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader

from src.utils.ipums_extract import load_ipums_from_pkl
from src.models.neural_net import NeuralNetwork, CustomDataset

from src.utils.plots import (
    plot_roc_curve,
    generate_confusion_matrix
)

# Load Data
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mozambique.pkl"
try:
    mig1_df, mig5_df = load_ipums_from_pkl(DATA_PATH)
except FileNotFoundError as fnfe:
    raise FileNotFoundError(f"Data not found at: {DATA_PATH} !") from fnfe

N_JOBS = 4
SEED = 5523
CV_RATIO = 0.20
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO
assert np.isclose(np.sum([TRAIN_RATIO, VAL_RATIO, TEST_RATIO]), 1.0)

X1 = mig1_df.drop(columns=['MIGRATE1'], inplace=False, axis=1)
y1 = mig1_df['MIGRATE1'].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,
                                                        test_size=TEST_RATIO,
                                                        random_state=SEED,
                                                        stratify=y1)

X5 = mig5_df.drop(columns=['MIGRATE5'], inplace=False, axis=1)
y5 = mig5_df['MIGRATE5'].values

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5,
                                                        test_size=TEST_RATIO,
                                                        random_state=SEED,
                                                        stratify=y5)

# Load Results and Models
RESULTS_PATH = PROJECT_ROOT / "exports/results.pkl"
MODELS_PATH = PROJECT_ROOT / "exports/models.pkl"

with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

forest_results = results['random_forest']
network_results = results['neural_network']

forest1 = models['random_forest1']
forest5 = models['random_forest5']
network1 = models['neural_network1']
network5 = models['neural_network5']


# Generate ROC Curves

# Random Forest
rf1_fpr, rf1_tpr, _ = forest_results['test1']['roc_curve']
rf5_fpr, rf5_tpr, _ = forest_results['test5']['roc_curve']
rf1_auc = forest_results['test1']['roc_auc']
rf5_auc = forest_results['test5']['roc_auc']
plot_roc_curve(rf1_tpr, rf1_fpr, rf1_auc,
               export=PROJECT_ROOT / "exports/forest_plots/roc1.png")
plot_roc_curve(rf5_tpr, rf5_fpr, rf5_auc,
               export=PROJECT_ROOT / "exports/forest_plots/roc5.png")

# Neural Network
nn1_fpr, nn1_tpr, _ = network_results['test1']['roc_curve']
nn5_fpr, nn5_tpr, _ = network_results['test5']['roc_curve']
nn1_auc = network_results['test1']['roc_auc']
nn5_auc = network_results['test5']['roc_auc']
plot_roc_curve(nn1_tpr, nn1_fpr, nn1_auc,
               export=PROJECT_ROOT / "exports/network_plots/roc1.png")
plot_roc_curve(nn5_tpr, nn5_fpr, nn5_auc,
               export=PROJECT_ROOT / "exports/network_plots/roc5.png")


# Generate Confusion Matrices

# Random Forest
y_pred1 = forest1.predict(X1_test)
y_pred5 = forest5.predict(X5_test)
generate_confusion_matrix(y1_test, y_pred1, labels=[0, 1],
                          export=PROJECT_ROOT / "exports/forest_plots/cm1.png")
generate_confusion_matrix(y5_test, y_pred5, labels=[0, 1],
                          export=PROJECT_ROOT / "exports/forest_plots/cm5.png")

# Neural Network
def get_predictions(model, X_test, y_test):
    """Compute predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            _, preds = torch.max(out, 1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_pred, y_true

y_pred1, y_true1 = get_predictions(network1, X1_test, y1_test)
y_pred5, y_true5 = get_predictions(network5, X5_test, y5_test)

generate_confusion_matrix(y_true1, y_pred1, labels=[0, 1],
                          export=PROJECT_ROOT / "exports/network_plots/cm1.png")
generate_confusion_matrix(y_true5, y_pred5, labels=[0, 1],
                          export=PROJECT_ROOT / "exports/network_plots/cm5.png")
