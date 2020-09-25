## Libraries
# General math and I/O modules.
import numpy as np
import pandas as pd
# Machine Learning library.
import sklearn
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

## Files
# Training file
train_filename = 'train.csv'
# Output file
output_filename = 'output.csv'

## Execution parameters
# Lambda vector
lambda_vec = np.array([0.01,0.1,1,10,100])
#Cross-validation folds
n_folds = 10

## Random seed
# Initialize random seed
seed = 1234

## Code
# K-fold cross-validation initializing
kf = KFold(n_folds,shuffle=True,random_state=seed)
# Shuffle enabled to improve randomness of the cross-validation folds.

# Import Pandas DataFrame from input file
train_set = pd.read_csv(train_filename).set_index("Id")

# Convert Pandas DataFrame to NumpyArray
y = train_set['y'].to_numpy()
X = train_set.loc[:,'x1':].to_numpy()

# Create list to store RMSEs values
rmse_list = []

# Iterate over lambda_vec
for lambda_value in lambda_vec:

    # Create temporary list
    rmse_temp = []
    # Initialize regressor
    reg = linear_model.Ridge(alpha=lambda_value,solver='saga',random_state=seed)
    # SAGA is the best choice for computational performance.

    for train_index, test_index in kf.split(X):

        # Split Trainig Set and Test Set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit
        reg.fit(X_train,y_train)
        # Predict
        y_pred = reg.predict(X_test)
        # Compute RMSE for the current test set and append
        rmse_temp.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Append the mean of the previous RMSEs
    rmse_list.append(np.average(np.array(rmse_temp)))

# Create an numpy array out of the list
rmse = np.array(rmse_list)
# Create Pandas Series
rmse_series = pd.Series(rmse)
# Output to file
rmse_series.to_csv(output_filename,header=False,index=False)
