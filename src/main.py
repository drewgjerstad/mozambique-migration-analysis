"""
This module serves as the "main" script to train, tune, and evaluate the three
classification models applied to the Mozambique dataset with the goal of
classifying census samples' migration status. While a notebook exists in the
main directory of this repo that does a similar job, the tuning is best run on
high-performance GPUs--such as those found on MSI.
"""

# Load Dependencies
from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.utils.ipums_extract import load_ipums_from_pkl
from src.models.train import train_sklearn_model
from src.models.eval import evaluate_sklearn_model
from src.models.neural_net import (
    train_neural_net,
    evaluate_neural_net
)

# Define Paths to Store Models and Results
EXPORTS_DIR = Path(__file__).parent / "exports"
RESULTS_PATH = EXPORTS_DIR / "results.pkl"
MODELS_PATH = EXPORTS_DIR / "models.pkl"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Storage in Pickle Files
with open(RESULTS_PATH, 'wb') as f:
    results = {'random_forest': {'validation1': None, 'test1': None,
                                 'validation5': None, 'test5': None},
               'support_vector': {'validation1': None, 'test1': None,
                                  'validation5': None, 'test5': None},
               'neural_network': {'validation1': None, 'test1': None,
                                  'validation5': None, 'test5': None}}
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
f.close()

with open(MODELS_PATH, 'wb') as f:
    models = {'random_forest1': None,
              'random_forest5': None,
              'support_vector1': None,
              'support_vector5': None,
              'neural_network1': None,
              'neural_network5': None}
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)
f.close()

# Define Hyperparameters
N_JOBS = 4
SEED = 5523
CV_RATIO = 0.20
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO
assert np.isclose(np.sum([TRAIN_RATIO, VAL_RATIO, TEST_RATIO]), 1.0)

# Define Hyperparameter Grids for Tuning
forest_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt'],
    'class_weight': ['balanced'],
    'bootstrap': [True],
    'random_state': [SEED]
}

svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale'],
    'class_weight': ['balanced'],
    'random_state': [SEED],
    'probability': [True]
}

# Define Hyperparameters for Neural Network
N_EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# Load Data
PROJECT_ROOT = Path(__file__).parent.parent
PKL_PATH = PROJECT_ROOT / "data" / "mozambique.pkl"
try:
    mig1_df, mig5_df = load_ipums_from_pkl(PKL_PATH)
    print(f"Loaded Mozambique migration datasets from {PKL_PATH} .")
    print(f"  Shape of mig1_df: {mig1_df.shape}")
    print(f"  Shape of mig5_df: {mig5_df.shape}")
except FileNotFoundError as fnfe:
    raise FileNotFoundError(f"Data file not found at: {PKL_PATH} !") from fnfe

# Create Development Splits (Train/Val/Test)
print("\nCreating development splits...")
X1 = mig1_df.drop(columns=['MIGRATE1'], inplace=False, axis=1)
y1 = mig1_df['MIGRATE1'].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,
                                                        test_size=TEST_RATIO,
                                                        random_state=SEED,
                                                        stratify=y1)

X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train,
                                                      test_size=VAL_RATIO/(1-TEST_RATIO),
                                                      random_state=SEED,
                                                      stratify=y1_train)

X5 = mig5_df.drop(columns=['MIGRATE5'], inplace=False, axis=1)
y5 = mig5_df['MIGRATE5'].values

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5,
                                                        test_size=TEST_RATIO,
                                                        random_state=SEED,
                                                        stratify=y5)

X5_train, X5_val, y5_train, y5_val = train_test_split(X5_train, y5_train,
                                                      test_size=VAL_RATIO/(1-TEST_RATIO),
                                                      random_state=SEED,
                                                      stratify=y5_train)

print("Created development splits for MIG1 and MIG5.")
print(f"  Shape of MIG1 train: {X1_train.shape}")
print(f"  Shape of MIG1 validation: {X1_val.shape}")
print(f"  Shape of MIG1 test: {X1_test.shape}")
print(f"  Shape of MIG5 train: {X5_train.shape}")
print(f"  Shape of MIG5 validation: {X5_val.shape}")
print(f"  Shape of MIG5 test: {X5_test.shape}")


# Create Training Samples for CV ===============================================
print("\nCreating random subsets of training data for cross-validation...")
X1_train_sample, _, y1_train_sample, _ = train_test_split(
    X1_train, y1_train, train_size=CV_RATIO, random_state=SEED, stratify=y1_train
)

X5_train_sample, _, y5_train_sample, _ = train_test_split(
    X5_train, y5_train, train_size=CV_RATIO, random_state=SEED, stratify=y5_train
)

print("Created random subsets of training data for cross-validation.")
print(f"  Shape of MIG1 train sample: {X1_train_sample.shape}")
print(f"  Shape of MIG5 train sample: {X5_train_sample.shape}")
# ==============================================================================

# Random Forest Classifier =====================================================
print("Starting random forest classifier...")

# Initialize classifiers
forest1 = RandomForestClassifier(n_jobs=N_JOBS)
forest5 = RandomForestClassifier(n_jobs=N_JOBS)

# Train and Evaluate for MIG1
print("  -> Tuning random forest hyperparameters on MIG1 training sample...")
forest1, forest1_results = train_sklearn_model(
    forest1, forest_param_grid, X1_train_sample, y1_train_sample, X1_val, y1_val)
print("  -> Tuning finished. Training final model on full MIG1 training set...")
best_forest_params1 = forest1.get_params()
forest1 = RandomForestClassifier(**{k: v for k, v in best_forest_params1.items()
                                    if k in forest_param_grid.keys()})
forest1.fit(X1_train, y1_train)
print("  -> Training finished. Evaluation started...")
forest1_test_results = evaluate_sklearn_model(forest1, X1_test, y1_test)
print("  -> Evaluation finished. Here are the results:")
print("       Validation Set (MIG1) Results:")
for metric_name, val in forest1_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")
print("       Test Set (MIG1) Results:")
for metric_name, val in forest1_test_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")

# Train and Evaluate for MIG5
print("\n  -> Tuning random forest hyperparameters on MIG5 training sample...")
forest5, forest5_results = train_sklearn_model(
    forest5, forest_param_grid, X5_train_sample, y5_train_sample, X5_val, y5_val)
print("  -> Tuning finished. Training final model on full MIG5 training set...")
best_forest_params5 = forest5.get_params()
forest5 = RandomForestClassifier(**{k: v for k, v in best_forest_params5.items()
                                    if k in forest_param_grid.keys()})
forest5.fit(X5_train, y5_train)
print("  -> Training finished. Evaluation started...")
forest5_test_results = evaluate_sklearn_model(forest5, X5_test, y5_test)
print("  -> Evaluation finished. Here are the results:")
print("       Validation Set (MIG5) Results:")
for metric_name, val in forest5_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")
print("       Test Set (MIG5) Results:")
for metric_name, val in forest5_test_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")

# Export Results
with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

results['random_forest']['validation1'] = forest1_results
results['random_forest']['test1'] = forest1_test_results
results['random_forest']['validation5'] = forest5_results
results['random_forest']['test5'] = forest5_test_results

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
print(f"Random forest results saved to {RESULTS_PATH} .")

with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

models['random_forest1'] = forest1
models['random_forest5'] = forest5

with open(MODELS_PATH, 'wb') as f:
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

print(f"Random forest classifiers saved to {MODELS_PATH} .")

print("Finished random forest classifier.")
# ==============================================================================

# Support Vector Classifier ====================================================
print("\nStarting support vector classifier...")

# Initialize classifiers
svc1 = SVC(max_iter=10_000)
svc5 = SVC(max_iter=10_000)

# Train and Evaluate for MIG1
print("  -> Tuning support vector hyperparameters on MIG1 training sample...")
svc1, svc1_results = train_sklearn_model(
    svc1, svc_param_grid, X1_train_sample, y1_train_sample, X1_val, y1_val)
print("  -> Tuning finished. Training model on full MIG1 training set...")
best_svc_params1 = svc1.get_params()
svc1 = SVC(**{k: v for k, v in best_svc_params1.items()
              if k in svc_param_grid.keys()})
svc1.fit(X1_train, y1_train)
print("  -> Training finished. Evaluation started...")
svc1_test_results = evaluate_sklearn_model(svc1, X1_test, y1_test)
print("  -> Evaluation finished. Here are the results:")
print("       Validation Set (MIG1) Results:")
for metric_name, val in svc1_results.items():
    print(f"        * {metric_name}: {val}")
print("       Test Set (MIG1) Results:")
for metric_name, val in svc1_test_results.items():
    print(f"        * {metric_name}: {val}")

# Train and Evaluate for MIG5
print("\n  -> Tuning support vector hyperparameters on MIG5 training sample...")
svc5, svc5_results = train_sklearn_model(
    svc5, svc_param_grid, X5_train_sample, y5_train_sample, X5_val, y5_val)
print("  -> Tuning finished. Training model on full MIG5 training set...")
best_svc_params5 = svc5.get_params()
svc5 = SVC(**{k: v for k, v in best_svc_params5.items()
              if k in svc_param_grid.keys()})
svc5.fit(X5_train, y5_train)
print("  -> Training finished. Evaluation started...")
svc5_test_results = evaluate_sklearn_model(svc5, X5_test, y5_test)
print("  -> Evaluation finished. Here are the results:")
print("       Validation Set (MIG5) Results:")
for metric_name, val in svc5_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")
print("       Test Set (MIG5) Results:")
for metric_name, val in svc5_test_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")

# Export Results
with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

results['support_vector']['validation1'] = svc1_results
results['support_vector']['test1'] = svc1_test_results
results['support_vector']['validation5'] = svc5_results
results['support_vector']['test5'] = svc5_test_results

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

print(f"Support vector classifier results saved to {RESULTS_PATH} .")

with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

models['support_vector1'] = svc1
models['support_vector5'] = svc5

with open(MODELS_PATH, 'wb') as f:
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

print(f"Support vector classifiers saved to {MODELS_PATH} .")

print("Finished support vector classifier.")
# ==============================================================================

# Neural Network ===============================================================
print("\nStarting neural network...")

# Train and Evaluate for MIG1
print("  -> Training started for neural network on MIG1...")
nn1, nn1_results = train_neural_net(X1_train, y1_train, X1_val, y1_val,
                                    N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
print("  -> Training finished. Evaluation started...")
nn1_test_results = evaluate_neural_net(nn1, X1_test, y1_test)
print("  -> Evaluation finished. Here are the results:")
print("       Test Set (MIG1) Results:")
for metric_name, val in nn1_test_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")

# Train and Evaluate for MIG5
print("  -> Training started for neural network on MIG5...")
nn5, nn5_results = train_neural_net(X5_train, y5_train, X5_val, y5_val,
                                    N_EPOCHS, BATCH_SIZE, LEARNING_RATE)
print("  -> Training finished. Evaluation started...")
nn5_test_results = evaluate_neural_net(nn5, X5_test, y5_test)
print("  -> Evaluation finished. Here are the results:")
print("       Test Set (MIG5) Results:")
for metric_name, val in nn5_test_results.items():
    if metric_name != 'roc_curve':
        print(f"        * {metric_name}: {val}")

# Export Results
with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

results['neural_network']['validation1'] = nn1_results
results['neural_network']['test1'] = nn1_test_results
results['neural_network']['validation5'] = nn5_results
results['neural_network']['test5'] = nn5_test_results

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

print(f"Neural networks results saved to {RESULTS_PATH} .")

with open(MODELS_PATH, 'rb') as f:
    models = pickle.load(f)

models['neural_network1'] = nn1
models['neural_network5'] = nn5

with open(MODELS_PATH, 'wb') as f:
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

print(f"Neural networks saved to {MODELS_PATH} .")

print("Finished neural network.")
# ==============================================================================

print("\nComplete.")
print(f"  -> Results saved to: {RESULTS_PATH}")
print(f"  -> Models saved to: {MODELS_PATH}")
