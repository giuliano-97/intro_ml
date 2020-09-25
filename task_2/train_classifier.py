## Libraries
# General purpose libraries
import pandas as pd
import numpy as np
from numpy import mean
import pickle
# Machine learning libraries
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# Import helpers
from helpers import *

## Execution parameters
# Number of classifications to preform
class_labels_no = 11
# Enable random forest cross-validation
cv_random_forest = False

## Functions
# Train random forest
def train_classifier_random_forest(X, y, label):
    y_selected = y[label].to_numpy()
    rf_clf = RandomForestClassifier(n_jobs=4, \
        criterion='gini', \
        max_depth=32, \
        min_samples_split=2, \
        min_samples_leaf=16, \
        random_state=seed, \
        class_weight='balanced', \
        n_estimators=200)
    if(cv_random_forest):
        scores = cross_val_score(rf_clf, X, y_selected, scoring='roc_auc', n_jobs=2)
        print(label)
        print("Random forest Roc auc score: %.2f%%" % mean(scores))

    rf_clf.fit(X, y_selected)
    return MedClassifierWrapper(label, rf_clf, \
            cross_validation_score=-1, \
            clf_params=None)

## Code
def main():

    # Load features classification (exclude pid)
    X = pd.read_csv(train_features_files[0]).to_numpy()[:,1:]

    # Load ground truth data
    Ys = pd.read_csv(train_labels_file)

    # Get selected features dictionary
    features_sel = get_model(train_features_sel_file)

    # Train a binary classifier for each class label
    medical_test_classifiers = []
    # Iterate over classification labels
    for label in Ys.columns[1:(class_labels_no+1)]:
        # Train and add classifier to list
        medical_test_classifiers.append(train_classifier_random_forest(X, Ys, label))

    # Save classifiers list
    pickle.dump(medical_test_classifiers, open(trained_models[0], 'wb'))

if __name__=='__main__':
    main()
