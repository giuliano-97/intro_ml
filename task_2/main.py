## Libraries
# Import custom modules
import feature_extraction
import feature_selection
import train_classifier
import train_regressor
import predict
import random

## Random seed
# Set random seeds
random.seed(seed)
np.random.seed(seed)

## Code
print('Feature extraction')
feature_extraction.main()
print('Feature selection')
feature_selection.main()
print('Training regressor')
train_regressor.main()
print('Training classifier')
train_classifier.main()
print('Predicting')
predict.main()
