## Libraries
# General purpose libraries
import os
import pickle
import random
import numpy as np
import pandas as pd
from zipfile import ZipFile
# Machine learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Lambda

## Execution parameters
# Shape of the resized images
img_shape = [299,299]
# Shape of the input tensor
input_shape = (299,299,3)
# Validation triplets
num_val_triplets = 1500
# Number of images used in the training set
num_images_training = 5000

## Random seed
# Initialize random seed
seed = 528
# Fix random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

## Files
# Zip archive of the images
#################################################################################
# The code automatically extracts images from food.zip to the directory ./food/ #
#################################################################################
images_archive = 'food.zip'
# Resized images directory
res_img_dir = './food_res/'
# Features file
features_file = 'features.pckl'
# Training set
train_triplets_file = 'train_triplets.txt'
# Test set
test_triplets_file = "test_triplets.txt"
# Submission file
submission_file = 'output.txt'

## Functions
# Resize the dataset images
def gen_resized_dataset(zip_file,res_img_dir,res_shape):

    # Only debug purposes
    print('Extracting .zip archive...')

    # Open .zip archive
    zip_ref = ZipFile(zip_file, 'r')
    # Extract food.zip to the zip directory
    zip_ref.extractall()
    # Get zip folder name
    img_dir = zip_ref.filename[:-4]
    # Close .zip archive
    zip_ref.close()

    # Only debug purposes
    print('Archive successfully extracted!')

    # Check whether res_img_dir exists
    if not os.path.exists(res_img_dir):
        # If not, create the directory
        os.makedirs(res_img_dir)

    # Only debug purposes
    print('Resizing images...')
    count = 0
    size = len(os.listdir(img_dir))

    # Iterate over all the images inside the img_dir
    for filename in os.listdir(img_dir):

        # Only debug purposes
        count += 1
        print('Processed files: {}/{}'.format(count,size),end="\r")

        # If the file is actually an image
        if filename.endswith('.jpg'):
            # Load image from the directory
            img = load_img(img_dir+'/'+filename)
            # Convert image to array
            img = img_to_array(img)
            # Resize the image according to img_shape
            img = tf.image.resize_with_pad(img,img_shape[0],img_shape[1],antialias=True)
            # Convert the array back to image
            img = array_to_img(img)
            # Save image to disk in the res_food dir
            img.save(res_img_dir+'/'+str(int(os.path.splitext(filename)[0]))+'.jpg')

    # Only debug purposes
    print('All images were successfully resized and saved to disk!')

# Generate the feature extraction neural network
def feature_extraction_net(input_shape):

    # resnet feature extraction
    resnet_inception = tf.keras.applications.InceptionResNetV2(pooling='avg',include_top=False)
    # restnet takes care of features extraction
    resnet_inception.trainable = False

    # Declare input
    x = x_in = Input(shape=input_shape)
    x = resnet_inception(x)

    # Get the whole model
    model = Model(inputs=x_in, outputs=x)

    return model

# Image generator
def image_from_directory_generator(directory_name, batch_size):
    # Image indices
    num_images = 10000
    # Current idx
    curr_idx = 0

    while True:
        batch = []

        while len(batch) < batch_size:
            img_name= directory_name + str(int(curr_idx)) + ".jpg"
            img = load_img(img_name)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(img))
            batch.append(img)
            curr_idx = (curr_idx + 1) % num_images

        batch = np.array(batch)
        labels = np.zeros(batch_size)

        try:
            yield batch, labels
        except StopIteration:
            return

# Perform feature extraction
def feature_extraction():
    # Declare feature selection model
    feature_extraction = feature_extraction_net(input_shape)

    # Initialize generator
    res_imgs = image_from_directory_generator(res_img_dir,1)

    # Compute features
    x_feat = feature_extraction.predict(res_imgs,steps=10000)

    # Return features vector
    return x_feat

# Create cross validation and training disjoint splits
def disjointSplit(full_train_file, train_disjoint_file, validation_disjoint_file):
    # Import the triplets tables
    train_trip_df = pd.read_csv(full_train_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])

    # Fix random seed
    random.seed(a=1891237)

    # Create a set of image ids and a list of triples indices
    val_img_ids = {random.sample(range(num_images_training), 1)[0]}
    val_trips_idxs = []
    # Start by adding 3 random numbers
    val_img_ids.update(random.sample(range(num_images_training) , 3))
    # Iteratively add image ids to the set until the set of images
    # containing only those ids
    while True:
        # Get all triplets such that their ids all belong within the current set
        ids = list(val_img_ids)
        val_trips_idxs = train_trip_df.index[train_trip_df.isin(ids).any(1)]
        curr_val_set_size = len(val_trips_idxs)
        if(curr_val_set_size >= num_val_triplets):
            break
        else:
            # Add another number to the set - pick it from the last set of triplet
            newSetIds = np.unique(train_trip_df.iloc[val_trips_idxs].to_numpy()).tolist()
            newId = random.choice(newSetIds)
            while(newId in val_img_ids):
                newId = random.choice(newSetIds)
            val_img_ids.add(newId)

    # Now that we have the validation triplets, let's get all the training triplets
    full_val_img_ids = np.unique(train_trip_df.iloc[val_trips_idxs].to_numpy()).tolist()
    train_trips_idxs = train_trip_df.index[~train_trip_df.isin(full_val_img_ids).any(1)]
    print("Number of training triplets: %d \nNumber of validation triplets: %d" %(len(train_trips_idxs),len(val_trips_idxs)))

    # Now save them to file
    train_trip_df.iloc[train_trips_idxs].to_csv(train_disjoint_file, sep=" ", index=False, header=False)
    train_trip_df.iloc[val_trips_idxs].to_csv(validation_disjoint_file, sep=" ", index=False, header=False)

# Create the input tensor of features
def buildTripletTensor(features, triplets_file, gen_labels=False):
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])
    # Features tensor
    train_tensors = []
    # Labels
    labels = []
    # Number of triplets in the file
    num_triplets = len(triplets_df)

    for i in range(num_triplets):
        # Get triplet
        triplet = triplets_df.iloc[i]
        Q, P, N = triplet['Q'], triplet['P'], triplet['N']
        # Get features
        tensor_q = features[Q]
        tensor_p = features[P]
        tensor_n = features[N]
        # Concatenete
        triplet_tensor = np.concatenate((tensor_q, tensor_p, tensor_n), axis=-1)
        if(gen_labels):
            reverse_triplet_tensor = np.concatenate((tensor_q, tensor_n, tensor_p), axis=-1)
            # Add to train tensors
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(reverse_triplet_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)


    train_tensors = np.array(train_tensors)
    if(gen_labels):
        labels = np.array(labels)
        return train_tensors, labels
    else:
        return train_tensors

## Code
def main():
    # Generate images
    gen_resized_dataset(images_archive, res_img_dir, img_shape)
    # Check if features have already been extracted
    if(os.path.exists(features_file)):
        # Load features
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        # Extract features with inception resnet
        print("Extracting features with pretrained model...")
        features = feature_extraction()
        # Save features in feature file
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)

    print("Features loaded!")

    print("Generating training and test features tensor...")
    # Get train tensors and labels
    train_tensors, labels = buildTripletTensor(features, 'train_triplets.txt', gen_labels=True)
    # Get test tensor
    test_tensors = buildTripletTensor(features, 'test_triplets.txt', gen_labels=False)
    print("Feature tensors generated!")

    print("Building model...")
    # Build model to process features
    x = x_in = Input(train_tensors.shape[1:])
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1152)(x)
    x = Activation('relu')(x)
    x = Dense(288)(x)
    x = Activation('relu')(x)
    x = Dense(72)(x)
    x = Activation('relu')(x)
    x = Dense(18)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x)
    print("Compiling model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(x = train_tensors, y = labels, epochs=4)
    print("Training completed!")

    # Predict
    print("Making inference...")
    y_test = model.predict(test_tensors)

    # Create submission file
    print("Genrating submission file...")
    y_test_thresh = np.where(y_test < 0.5, 0, 1)
    np.savetxt(submission_file, y_test_thresh, fmt='%d')
    print("Submission file generated! Done.")

if __name__ == '__main__':
    main()
