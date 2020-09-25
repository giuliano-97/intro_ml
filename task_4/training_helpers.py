import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def siamese_image_generator(triplets_file, data_dir, batch_size, leading_zeros=False):
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])
    # Number of triplets
    num_triplets = len(triplets_df)
    # Index in the data frame of the next triplet
    next_triplet_idx = 0

    # Start looping
    while True:
        # Initialize empty lists of images
        batch = []

        # Fill the batch
        while len(batch) < batch_size:
            # Get new triplet - it will be a pandas Series
            triplet = triplets_df.iloc[next_triplet_idx]
            # Increase triplet count but make sure not to go out of bound
            next_triplet_idx = (next_triplet_idx + 1) % num_triplets
            # Now get the number of each image in the triplet
            # Q = Query image, P = Positive, N = Negative
            Q, P, N = triplet['Q'], triplet['P'], triplet['N']
            if(leading_zeros):
                Q_img_file_name = ("%05d" % (Q,)) + ".jpg"
                P_img_file_name = ("%05d" % (P,)) + ".jpg"
                N_img_file_name = ("%05d" % (N,)) + ".jpg"                
            else:
                Q_img_file_name = str(Q) + ".jpg"
                P_img_file_name = str(P) + ".jpg"
                N_img_file_name = str(N) + ".jpg"
            # Now load the images from the image directory and resize them
            Q_img = load_img(data_dir + Q_img_file_name)
            P_img = load_img(data_dir + P_img_file_name)
            N_img = load_img(data_dir + N_img_file_name)
            Q_norm = img_to_array(Q_img) / 255
            P_norm = img_to_array(P_img) / 255
            N_norm = img_to_array(N_img) / 255
            # Add the triplet of images to the batch (in sequence)
            # batch.append(np.concatenate((Q_norm, P_norm, N_norm)))
            batch.append(np.array([Q_norm, P_norm, N_norm]))

        # Empty labels - we don't need labels for this task
        labels = np.zeros(batch_size)

        # Convert to numpy array
        batch = np.array(batch)

        # Now transform the list into an array - this will be the next batch
        try:
            yield batch, labels
        except StopIteration:
            return


def image_from_directory_generator(directory_name, batch_size):
    # Image indices
    num_images = 10000
    # Current idx
    curr_idx = 0
    # 
    

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

def nine_channels_image_generator(triplets_file, data_dir, batch_size, leading_zeros=False, shuffle=False):
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N", "Label"])
    # Number of triplets
    num_triplets = len(triplets_df)
    # Position in the index list of the index of the next row
    next_triplet_idx = 0
    # Dataframe index
    df_index = triplets_df.index
    df_index_numpy = np.array(triplets_df.index)
    np.random.shuffle(df_index_numpy)
    df_index = df_index_numpy.tolist()

    

    # Start looping
    while True:
        # Initialize empty lists of images
        batch = []
        labels = []

        # Fill the batch
        while len(batch) < batch_size:
            # Get new triplet - it will be a pandas Series
            triplet = triplets_df.iloc[df_index[next_triplet_idx]]
            # Increase triplet count but make sure not to go out of bound
            next_triplet_idx = (next_triplet_idx + 1) % num_triplets
            if(next_triplet_idx == 0 and shuffle):
                # Shuffle index
                np.random.shuffle(df_index_numpy)
                df_index = df_index_numpy.tolist()
                # Reindex dataframe
                triplets_df = triplets_df.reindex(df_index)
            # Now get the number of each image in the triplet
            # Q = Query image, P = Positive, N = Negative
            Q, P, N, label = triplet['Q'], triplet['P'], triplet['N'], triplet['Label']
            if(leading_zeros):
                Q_img_file_name = ("%05d" % (Q,)) + ".jpg"
                P_img_file_name = ("%05d" % (P,)) + ".jpg"
                N_img_file_name = ("%05d" % (N,)) + ".jpg"                
            else:
                Q_img_file_name = str(int(Q)) + ".jpg"
                P_img_file_name = str(int(P)) + ".jpg"
                N_img_file_name = str(int(N)) + ".jpg"
            # Now load the images from the image directory and resize them
            Q_img = load_img(data_dir + Q_img_file_name)
            P_img = load_img(data_dir + P_img_file_name)
            N_img = load_img(data_dir + N_img_file_name)
            Q_norm = img_to_array(Q_img) / 255
            P_norm = img_to_array(P_img) / 255
            N_norm = img_to_array(N_img) / 255
            # Add the triplet of images to the batch (in sequence)
            batch.append(np.array([Q_norm, P_norm, N_norm]))
            labels.append(label)

        # Empty labels - we don't need labels for this task
        labels = np.array(labels)

        # Convert to numpy array
        batch = np.array(batch)

        # Now transform the list into an array - this will be the next batch
        try:
            yield batch, labels
        except StopIteration:
            return

def buildTripletTensor(features, triplets_file):
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])
    # Features tensor
    train_tensors = []
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
        # Add to train tensors
        train_tensors.append(triplet_tensor)

    train_tensors = np.array(train_tensors)

    return train_tensors


def buildTripletTensorWithLabels(features, triplets_file):
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
        reverse_triplet_tensor = np.concatenate((tensor_q, tensor_n, tensor_p), axis=-1)
        # Add to train tensors
        train_tensors.append(triplet_tensor)
        labels.append(1)
        train_tensors.append(reverse_triplet_tensor)
        labels.append(0)


    train_tensors = np.array(train_tensors)
    labels = np.array(labels)

    return train_tensors, labels

        

