## Libraries
# General purpose libraries
import pandas as pd
import numpy as np
import pickle
# Machine Learning libraries
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
# Import helpers
from helpers import *

## Executon parameter
output_labels_ordered = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate', 'LABEL_ABPm',
       'LABEL_SpO2', 'LABEL_Heartrate']

## Functions
# Sigmoid
def sigmoid(value):
    return 1.0/(1+np.exp(-value))

## Code
def main():
    # Predictions list
    predictions = []

    # Get selected features dictionary
    features_sel = get_model(train_features_sel_file)

    # Predict classification
    # Load features classification (exclude pid)
    X_class = pd.read_csv(test_features_files[0]).to_numpy()[:,1:]

    # Load trainded models
    medical_test_classifiers = get_model(trained_models[0])

    # Iterate over classifiers
    for wrapper in medical_test_classifiers:
        # Predict classifier
        y = wrapper.clf_.predict_proba(X_class)[:,1]
        predictions.append(pd.DataFrame(y,columns=[wrapper.label_]))

    # Predict regression
    # Load features regression (exclude pid)
    X_rgr = pd.read_csv(test_features_files[1]).to_numpy()[:,1:]

    # Load scaler for regression features
    rgr_scaler_ftr = get_model(rgr_scaler_ftr_file)

    # Load scalers for regression labels
    rgr_scalers_lbl = get_model(rgr_scalers_lbl_file)

    # Standardize features
    X_rgr_scaled = rgr_scaler_ftr.transform(X_rgr)

    # Load trained models
    medical_params_regressors = get_model(trained_models[1])

    # Predict regressor
    for wrapper in medical_params_regressors:
        # Get labels regression scaler
        rgr_scaler_lbl = rgr_scalers_lbl[wrapper.label_]
        # Predict standardized values
        y_std = wrapper.best_estimator_.predict(X_rgr_scaled[:,features_sel[wrapper.label_]])
        # De-standardize values
        y = rgr_scaler_lbl.inverse_transform(y_std)
        predictions.append(pd.DataFrame(y,columns=[wrapper.label_]))

    # Generate output file
    # Get pid from first column of features files
    output = pd.read_csv(test_features_files[1])['pid']
    # Build results tables to ouput
    for label in output_labels_ordered:
        to_append = pd.DataFrame(np.zeros(X_rgr.shape[0]),columns=[label])
        for prediction in predictions:
            if(prediction.columns[0]==label):
                to_append = prediction
                break
        output = pd.concat([output, to_append],axis=1)

    # Write the output file
    compression_opts = dict(method='zip',archive_name=prediction_file)
    output.to_csv(prediction_archive, index=False, float_format='%.3f', compression=compression_opts)

if __name__=='__main__':
    main()
