"""
## Trains the classifier and saves the weights
"""

import os
import numpy as np
from scipy import misc
from glob import glob
from utils import py_utils, data_utils, utilities_module_1
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping


"""
## Find the images
"""
im_dir = os.path.join('data', 'module_1')
im_ext = '.bmp'
files = np.asarray(glob(os.path.join(im_dir, '*%s' % im_ext)))

"""
## Load the images
"""
ims = []
for f in files:
    ims += [misc.imread(f)]
ims = np.asarray(ims)
num_ims, height, width, channels = ims.shape
py_utils.status(
    'Found %s cat-dog images with size %s x %s x %s pixels' % (
        num_ims, height, width, channels))

"""
## Split images into cats and dogs.
"""
labels = utilities_module_1.derive_labels(files)
files = files[labels != 0]  # Remove "template" images
ims = ims[labels != 0]
labels = labels[labels != 0]
labels -= 1
num_ims = len(labels)

"""
## Reshape ims into a (example) x (pixel) matrix
"""
ims = ims.reshape(num_ims, height * width * channels)

"""
## Split files into train and test sets
"""
training_split = 0.9  # Hold out this proportion for training
dog_labels = np.where(labels == 1)[0]  # Balance categories in each split
cat_labels = np.where(labels == 2)[0]  # Balance categories in each split
dog_idx = np.random.permutation(len(dog_labels))
cat_idx = np.random.permutation(len(cat_labels))
dog_training_idx = dog_idx[:int(len(dog_idx) * training_split)]
dog_testing_idx = dog_idx[int(len(dog_idx) * training_split):]
cat_training_idx = cat_idx[:int(len(cat_idx) * training_split)]
cat_testing_idx = cat_idx[int(len(cat_idx) * training_split):]
combined_training = np.hstack((dog_training_idx, cat_training_idx))
combined_testing = np.hstack((dog_testing_idx, cat_testing_idx))
training_ims = ims[combined_training]
training_labels = labels[combined_training]
training_files = files[combined_training]
testing_ims = ims[combined_testing]
testing_labels = labels[combined_testing]
testing_files = files[combined_testing]

"""
## Normalize data TODO: WHY DO YOU NORMALIZE?
"""
norm_train_ims, train_mu, train_sd = data_utils.normalize(
    training_ims,
    method='zscore')

"""
## Build and train a logistic classifier with keras.
"""
py_utils.status(
    'Building logistic classifer. Training on cat-dog data.')
model = Sequential()
model.add(
    Dense(
        1,
        activation='sigmoid',
        input_dim=height * width * channels))
optim = optimizers.adam()
# optim = optimizers.SGD(
#     lr=0.01,
#     momentum=0.0,
#     decay=0.0,
#     nesterov=True)
model.compile(
    optimizer=optim,
    loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
losses = model.fit(  # TODO P1 OF ASSIGNMENT SHOULD LOAD THESE WEIGHTS
    norm_train_ims,
    training_labels,
    epochs=2000,
    callbacks=[early_stopping],
    validation_split=0.1)

model_weights = model.get_weights()
np.save(os.path.join('models', 'weights'), model_weights)
np.save(os.path.join('models', 'test_indices'), combined_testing)
np.save(os.path.join('models', 'train_mu'), train_mu)
np.save(os.path.join('models', 'train_sd'), train_sd)
