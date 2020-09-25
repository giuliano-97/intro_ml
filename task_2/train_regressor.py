## Libraries
# General purpose libraries
import pandas as pd
import numpy as np
import pickle
# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import rbf_kernel
# Import helpers
from helpers import *

## Executon parameters
# Number of regressions to preform
regression_labels_no = 4
# Enable grid search on hyperparameters
grid_search = False

## Functions
# Train regressor
def train_select_regressor(X, y, param_grid, label, scalers_dict):
    # Select label
    y_selected = y[label].to_numpy()
    # Standardize y
    y_selected_std = scalers_dict[label].transform(y_selected.reshape(-1, 1))
    # Initialize regressor
    if (grid_search):
        # Instantiate model
        kern_regr = KernelRidge(kernel="rbf")
        # Initialize Grid Search
        reg = GridSearchCV(kern_regr, param_grid, verbose=3, n_jobs=2, scoring='r2')
        # Refit
        reg.fit(X,y_selected_std)
        # Return regressor wrapper
        return MedRegressorWrapper(label, reg.best_estimator_, reg.best_params_, reg.best_score_)
    else:
        # Instantiate model
        kern_regr = KernelRidge(kernel="rbf", alpha=1, gamma=0.01)
        # Fit
        kern_regr.fit(X, y_selected_std)
        # Return regressor
        return MedRegressorWrapper(label, kern_regr, None, -1)

# Cross-validation parameters dictionary generator
def genKernelParamsDict():
    kernel_params_dicts = []
    gammas = [0.01, 0.1, 1]
    for gamma in gammas:
        kernel_params_dicts.append({'gamma':gamma})

    return kernel_params_dicts

## Code
def main():

    # Define param grid for CV search
    param_grid = {"alpha": [0.1,1,10], "kernel_params":genKernelParamsDict()}

    # Load features regression (exclude pid)
    X = pd.read_csv(train_features_files[1]).to_numpy()[:,1:]

    # Load scaler for regression features
    rgr_scaler_ftr = get_model(rgr_scaler_ftr_file)

    # Load ground truth data for regression
    Ys = pd.read_csv(train_labels_file)

    # Load scaler dictionary for regression labels
    rgr_scalers_lbl = get_model(rgr_scalers_lbl_file)

    # Standardize features
    X_scaled = rgr_scaler_ftr.transform(X)

    # Get selected features dictionary
    features_sel = get_model(train_features_sel_file)

    # Train a regressor for each class label
    medical_params_regressors = []

    # Iterate over regression labels
    for label in Ys.columns[-regression_labels_no:]:

        # Train and add classifier to list
        medical_params_regressors.append(train_select_regressor(X_scaled[:,features_sel[label]], Ys, param_grid, label, rgr_scalers_lbl))

    # Save regressors dict
    pickle.dump(medical_params_regressors, open(trained_models[1], 'wb'))

if __name__ == "__main__":
    main()
