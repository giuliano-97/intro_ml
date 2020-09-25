## Libraries
# General purpose libraries
import pandas as pd
import numpy as np
import random
# Machine learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

## Files
# Training data file
train_filename = 'train.csv'
# Test data file
test_filename = 'test.csv'
# Output file
output_filename = 'output.csv'

## Execution parameters
# Batch size
batch_size = 64
# Epochs
epochs = 30

## Random seed
# Initialize random seed
seed = 528
# Set random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

## Functions
# f1 score definition
def f1(y_true, y_pred):

    # recall definition
    def recall(y_true, y_pred):
        #Recall metric.
        #Only computes a batch-wise average of recall.
        #Computes the recall, a metric for multi-label classification of
        #how many relevant items are selected.
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    # precision definition
    def precision(y_true, y_pred):
        #Precision metric.
        #Only computes a batch-wise average of precision.
        #Computes the precision, a metric for multi-label classification of
        #how many selected items are relevant.
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    # Compute precision
    precision = precision(y_true, y_pred)
    # Compute recall
    recall = recall(y_true, y_pred)
    # Return f1 score
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Create amino dict
def get_dict():
    # List of all aminacids
    amino_list = ['R','H','K',
                  'D','E','S',
                  'T','N','Q',
                  'C','U','G',
                  'P','A','I',
                  'L','M','F',
                  'W','Y','V']
    # Create empty dictionary
    amino_dict = {}
    # Get number of aminos
    length = len(amino_list)
    # Iterate over aminos
    for i in range(length):
        # Create zero amino_arrays
        vec = [0] * length
        # Assign 1 to the amino's position
        vec[i] = 1
        # Add amino_array to dict
        amino_dict[amino_list[i]] = vec
    # Return dict
    return amino_dict

# Convert to numpy
def seq_to_numpy(sequences):
    # Create dictionary of aminos
    amino_dict = get_dict()
    # Create empty list of amino_arrays
    vec_list = []
    # Iterate over all the sequences
    for seq in sequences:
        # Create empty temporary list
        temp_list = []
        # Iterate over every char
        for char in seq:
            # Extend amino_array to list
            temp_list.extend(amino_dict[char])
        # Append amino_matrix
        vec_list.append(temp_list)
    # Return numpy array
    return np.array(vec_list)

## Code
# Import Pandas DataFrame from input file
train_set = pd.read_csv(train_filename)

# Import Pandas DataFrame from input file
test_seq = pd.read_csv(test_filename)['Sequence']

# Gather labels
y_train = train_set['Active'].to_numpy()

# Gather sequences
train_seq = train_set['Sequence']

# Get train data
X_train = seq_to_numpy(train_seq)

# Get test data
X_test = seq_to_numpy(test_seq)

# Model declaration
model = Sequential()
model.add(Dense(84,input_shape=X_train[0].shape))
model.add(Activation('relu'))
model.add(Dense(42))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(21))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])

# Fit model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

# Predict
y_test = model.predict_classes(X_test)

# Output to file
pd.Series(y_test.reshape(-1)).to_csv(output_filename, header = False, index = False, index_label = False)
