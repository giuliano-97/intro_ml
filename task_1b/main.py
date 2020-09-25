## Libraries
#General math and I/O modules.
import numpy as np
import pandas as pd
# Machine Learning library.
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

## Files
# Training file
train_filename = 'train.csv'
# Output file
output_filename = 'output.csv'

## Execution parameters
# Lambda vector
lambda_vec = np.array([0.01,0.1,1,10,100])
# Cross-validation folds
n_folds = 10

## Random seed
# Initialize random seed
seed = 1234

## Functions
# Feature transform function
def feature_transform(X):

    # Concatenate all features except for the 21st which is the intercept
    X_trans = np.concatenate((X,np.power(X,2),np.exp(X),np.cos(X)),axis=1)
    # Return feature transformation matrix
    return X_trans

## Code
# Initialize regressor
reg = LassoCV(alphas=lambda_vec,cv=n_folds,selection='random',fit_intercept=False,random_state=seed)
# Initialize scaler
scaler = StandardScaler()

# Import Pandas DataFrame from input file
train_set = pd.read_csv(train_filename).set_index("Id")
# Convert Pandas DataFrame to NumpyArray
y = train_set['y'].to_numpy()
X_original = train_set.loc[:,'x1':].to_numpy()

# Transform features
X_trans = feature_transform(X_original)

# Standardize X
X_std = scaler.fit_transform(X_trans)
# Standardize y
y_mean = np.mean(y)
y_sigma = np.std(y)
y_std = np.divide((y-y_mean),y_sigma)

# Fit
reg.fit(X_std,y_std)

# Destandardize coefs up to feature 20
coef_destd_list = []
for coef,scale in zip(reg.coef_,scaler.scale_):
    coef_destd_list.append((coef/scale)*y_sigma)

# Destandardize intercept (mean of y values)
intercept_destd = y_mean - np.sum(np.multiply(np.multiply(np.divide(scaler.mean_,scaler.scale_),reg.coef_),y_sigma))

# Add intercept to list of coefficients
coef_destd_list.append(intercept_destd)
# Create an numpy array out of the list
coefs = np.array(coef_destd_list)

# Create Pandas Series
coefs_series = pd.Series(coefs)
# Output to file
coefs_series.to_csv(output_filename,header=False,index=False)
