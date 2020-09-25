# Task 4

This task involved training a simple deep neural network to determine, given three images of foods, which food between the first and the second image is more similar in taste to the third one.

This task could be solved very well with complex models using triplet loss, or siamese networks. However, for the sake of this project we used a very simple approach which turned out to work reasonably well and get us over the hard deadline for the IML project.

The training data is made up of triplets of images with ground truth (0 if the third image is more similar to the first and 1 if it is more similar to the second). We doubled the sized of the training dataset by adding for each triplet a new one with the first and the second image inverted and with a ground truth value opposite to the original one. 

For each triplet, we extracted a set vector of features from all three images using a pre-trained model (we used one of the Inception models available in Keras), concatenate these vectors and pass the resulting one to a fully connected NN as if it were a binary classification problem.




