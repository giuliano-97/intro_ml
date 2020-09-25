# Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
# Import helpers
from helpers import *

# Variables
k_best_rgr = 10
k_best_spo2 = 30
class_labels_no = 11
regression_labels_no = 4

def main():

    # Regression
    # Load labels
    Y = X_lbl = pd.read_csv(train_labels_file)

    # Initialize features_sel dict
    features_sel = {}

    # Load features
    X_rgr_train_ftr_pd = pd.read_csv(train_features_files[1])

    # Initialize scalers for regression features
    rgr_scaler_ftr = StandardScaler()

    # Standardize features
    X_rgr_train_ftr_std = rgr_scaler_ftr.fit_transform(X_rgr_train_ftr_pd.to_numpy()[:,1:])

    # Standardize labels
    # Create regression labels scalers list
    rgr_scalers_lbl = {}

    # Create scaled Ys list
    Y_std = {}

    # Compute scaled Ys
    for label in Y.columns[-regression_labels_no:]:
            rgr_scaler_lbl = StandardScaler()
            Y_std[label] = rgr_scaler_lbl.fit_transform(Y[label].to_numpy().reshape(-1, 1))[:,0]
            rgr_scalers_lbl[label] = rgr_scaler_lbl

    # Select k best for each regressor
    for label in Y.columns[-regression_labels_no:]:
        if label == 'LABEL_SpO2':
            selector = SelectKBest(mutual_info_regression, k_best_spo2)
            selector.fit(X_rgr_train_ftr_std, Y_std[label])
            features_sel[label] = selector.get_support()
        else:
            selector = SelectKBest(mutual_info_regression, k_best_rgr)
            selector.fit(X_rgr_train_ftr_std, Y_std[label])
            features_sel[label] = selector.get_support()

    # Output feature selection dictionary
    pickle.dump(features_sel, open(train_features_sel_file, 'wb'))

    # Output scaler for regression features
    pickle.dump(rgr_scaler_ftr, open(rgr_scaler_ftr_file, 'wb'))

    # Output scalers dictionary for regression labels
    pickle.dump(rgr_scalers_lbl, open(rgr_scalers_lbl_file, 'wb'))

if __name__=='__main__':
    main()
