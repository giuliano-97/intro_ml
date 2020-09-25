## Task 2

This task was mostly about missing data and classification.

The training data includes the results of a collection of medical tests performed at different times during the first 12 hours since being hospitalized of about 10000 patients, plus a set of binary labels denoting whether the patients turned up to be affected by a certain patology in the following 12 hours, as well as the values of certain health indicators (such as average blood pressure etc.).

The goal of this task was to: 
 1. Train a (set of) classification model(s) capable of predicting, given the results of medical tests performed during the first 12 hours of hospitalization, whether a patient would develop symptoms of any of the patologies listed in the training data.
 2. Train a (set of) regression model(s) capable of predicting the values of the health indicators listed in the training data.


Furthermore, since most medical tests are not performed every hour or for all the hospitalized patients, the training data is highly unbalanced and certain values are often missing.

To cope with the sparse nature of the training data, we computed a collection of summary statistics of the available features for every patient (e.g. empirical mean, stddev, and mean of the blood pressure during the first 12 hours of hospitalization with the available values) and for very sparse data (i.e. medical tests which are done 1 or 2 times, if at all during the first 12 hours) we used the number of missing values (count of "Nan" so to speak) as a feature.

After testing difference approaches through cross-validation, we obtained the best results using Random Forests for classification and LASSO for regression.

