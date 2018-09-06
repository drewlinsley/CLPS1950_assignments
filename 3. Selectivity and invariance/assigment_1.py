"""
## Module 3 assignment 1: Image filtering with convolutions


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 1, 2, and 3. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_3_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_3_assignment_1.zip

## Description

Convolutions are a powerful tool for measuring the similarity between a filter
and an image. Depending on how it is specified, a convolutional filter can
refine the look of an image -- much like the image filters available on
instagram -- or even be used to classify the content of images. Here you will
experiment with the latter implementation, which is one of the key elements in
recent and dramatic developments in computer vision and artificial
intelligence.

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 1 hour.

## Tips

If you are using Anaconda and Sublime, turn on the python linter.
    This will notify you of errors in your code.
    a. Click Sublime text -> Preferences -> Package settings -> Anaconda ->
        Settings - Default
    b. On line 310 set: "anaconda_linter_phantoms": true,
    c. Restart sublime 3.
If you are using a linter, your text editor will provide feedback on:
    a. Syntax errors
    b. Style.
    (a) feedback is critical to fix, but feel free to ignore (b) if it doesn't
    bother you. That said, it is important to maintain a consistent style as
    you code. This will reduce errors and make it easier for others to read.
If you have a GPU and CUDA installed:
    Make sure you "mask" your GPUs before running. Tensorflow (the backend of
    Keras) will automatically spread to all available GPUs, reserving as much
    GPU memory as possible. This means you must mask out GPUs that you don't
    want Tensorflow to see and spread to when running analyses with GPU.

    Do this by running python or ipython in the following way:

    CUDA_VISIBLE_DEVICES=0 python my_script.py

    or

    CUDA_VISIBLE_DEVICES=0 ipython

    where the number 0 refers to the only GPU you want visible for Tensorflow.
    Use the command nvidia-smi to see the available GPUs and their IDs, then
    exchange the 0 above for whatever GPU ID you'd like. You can make multiple
    GPUs visible by replacing the 0 with 0,1 or make no GPUs visible by setting
    a large positive or negative number (e.g. 999).

"""

import os
import numpy as np
from scipy.signal import convolve
from utils import utilities_module_3
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

"""
###########################
1. EDIT HERE!
###########################
Load a set of convolutional filters, visualize them, and apply them on images.

First create a loop that convolves the filters with an image and records an
activity at every location.

Next, perform the same analysis using the provided scipy function.

Q1. What do the convolutional filters look like? When you convolve a few of the
filters with an image, what kinds of visual patterns do they emphasize?

Estimated time: 15 minutes.
"""


"""
## Load and plot the filters.
"""

norm_filters, filters, biases = utilities_module_3.load_normalized_filters(
    path=os.path.join(
        'data',
        'alexnet.npy'))

# Visualize the normalized filters
utilities_module_3.plot_mosaic(
    norm_filters,
    title='Filters',
    rows=8,
    columns=12)

"""
## Create a function to convolve the first image with the filter bank
using the scipy convolve function.
"""
mr_roboto = utilities_module_3.proc_and_load_images(
    os.path.join('data', 'mr_roboto.jpg'))


def convolve_image(image, filters):
    """Convolve the image with the filters separately for each channel.

    Parameters
    ----------
    image : float
        [h, w, c] numpy image
    filters : float
        [n, fh, fw, k] numpy filter bank

    Returns
    -------
    numpy array
        A tensor of normalized activities

    """
    activities = []
    assert image.shape[-1] == filters.shape[-1],\
        'Image and filters need the same number of channels.'
    for f in filters:
        channel_activity = []
        for k in range(image.shape[-1]):
            conv = convolve(, , mode='same')
            channel_activity += [np.expand_dims(conv, axis=-1)]
        activities += [np.concatenate(channel_activity, axis=-1)]
    activities = np.asarray(activities)
    act_shape = activities.shape
    normalized_activities = utilities_module_3.normalize_to_zero_one(
        utilities_module_3.unravel(
            activities))
    reshaped_activities = utilities_module_3.ravel(normalized_activities, shape=act_shape)
    return reshaped_activities


# Let's look at a single color channel
chosen_filters = np.random.permutation(len(filters))[:10]
slice_mr_roboto = np.expand_dims(mr_roboto[:, :, 0], axis=-1)
select_filters = np.expand_dims(filters[chosen_filters, :, :, 0], axis=-1)
scipy_activities = convolve_image(
    image=,
    filters=)
utilities_module_3.plot_mosaic(
    scipy_activities,
    title='Scipy convolution activities',
    rows=2,
    columns=5)

"""
###########################
2. EDIT HERE!
###########################
Load CIFAR images and train two models to recognize them. One will be a
multilayer perceptron (as in module 2) and the other will apply the filters
that you analyzed in (1) via convolutions to encode and recognize images.

Q2. Create a plot comparing the performance of the two models that you have
trained. In words, describe the difference between the tolerances of your
model that uses convolutions to the multilayer perceptron. Include a brief
discussion of the number of parameters in each model, and the strategy that
multilayer perceptrons must use to solve visual recognition tasks.

Estimated time: 45 minutes.
"""

"""
## Load the images.
"""

x_train, y_train, x_test, y_test, _ = utilities_module_3.unpack_cifar()
# To make computation more efficient, we are going to limit the training and
# test sets to their first 250 images across 4 image categories.
x_train, y_train, keep_cats = utilities_module_3.trim_categories(
    x=x_train,
    y=y_train,
    n=250,
    cats=4)
x_test, y_test, _ = utilities_module_3.trim_categories(
    x=x_test,
    y=y_test,
    n=250,
    keep_cats=keep_cats)

# Process x_train and x_test datasets to match the dataset filters was
# trained on.
normalized_x_train = utilities_module_3.normalize_ims(x_train)
normalized_x_test = utilities_module_3.normalize_ims(x_test)

# One-hot encode labels and create a data generator
ey_train =
ey_test =
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False)
datagen.fit()  # Fit the datagen.
number, height, width, channels =
epochs =
output_dim =
lr = 3e-4
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)  # Note that we use ADAM, which is more
# flexible than SGD.

"""
## Create a multilayer perceptron that has a hidden layer with the same
number of parameters as the filters you analyzed above.
"""
num_filts, filt_height, filt_width, filt_channels = filters.shape
num_conv_params =
mlp_model = Sequential()
mlp_model.add(
    Reshape(  # Reshape an example from a tensor to a vector
        (height * width * channels,),
        input_shape=(height, width, channels)))
mlp_model.add(
    Dense(
        num_conv_params,
        activation='relu'))
mlp_model.add(
    Dense(
        output_dim,
        activation='softmax'))
mlp_model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])

# Train the dense model
mlp_output = mlp_model.fit_generator(
    datagen.flow(
        ,
        ,
        batch_size=batch_size),
    validation_data=datagen.flow(
        ,
        ,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
mlp_train_loss = mlp_output.history['loss']
mlp_test_loss = mlp_output.history['val_loss']

"""
## Initialize a convolutional model.
"""
conv_model = Sequential()
conv_model.add(
    Conv2D(
        ,  # Enter the number of filters here
        (filt_height, filt_width),
        activation='relu',
        input_shape=(height, width, channels)))
conv_model.add(
    Flatten())
conv_model.add(
    Dense(
        ,
        activation='softmax'))
conv_model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])

"""
## Replace the random convolutional weights provided by Keras with the filters
and biases you loaded above.
"""
conv_model_weights = conv_model.get_weights()
conv_model_weights[0] = filters.transpose(1, 2, 3, 0)
conv_model_weights[1] = biases
conv_model.set_weights(conv_model_weights)

# Train the convolutional model
conv_output = conv_model.fit_generator(
    datagen.flow(  # Grade
        ,
        ,
        batch_size=batch_size),
    validation_data=datagen.flow(
        ,
        ,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
conv_train_loss = conv_output.history['loss']
conv_test_loss = conv_output.history['val_loss']

# Plot performance of the MLP versus the convolutional filter model
f, ax = utilities_module_3.plot_performance(
    ,
    ,
    colors=['b', 'r'],
    label='MLP')
utilities_module_3.plot_performance(
    ,
    ,
    colors=['g', 'k'],
    label='Convolution',
    f=f,
    ax=ax)
plt.savefig('module_3_assigment_1_Q2_figure.png')
plt.show()
plt.close(f)


"""
###########################
3. EDIT HERE!
###########################
Why are convolutions important?

Build a function that "wraps" images by a specified number of pixels, shifting
image content either vertically or horizontally and wrapping pixels around to
the other side of the image. For instance, wrapping an image by one column
will move the its last column to its first, its first to its second, and so on.

Apply this function to the testing images and record the accuracy of the
convolutional and MLP models. Repeat the procedure when wraping 1-5 pixels.

Q3: Create a plot of the impact of wrapping pixels in CIFAR images vertically
(i.e. row-wise) on classifer performance of the MLP and convolutional models.

Q4: Repeat this for wraps in the horizontal direction (i.e. column-wise).
Describe your results and what this says about convolutional versus MLP models.

Estimated time: 15 minutes.
"""


def wrap_image(image, shift, axis=0):
    """Wrap the image about an axis.

    Parameters
    ----------
    image : float
        [h, w, c] numpy image
    shift : int
        Number of pixels to wrap the image
    axis : int
        Axis for wrapping

    Returns
    -------
    numpy array
        A shifted image

    """
    return np.roll(image, shift=shift, axis=axis)


def apply_wraps(images, shift, axis=0):
    """Apply the wrap operation to many images.

    Parameters
    ----------
    image : float
        [n, h, w, c] numpy images
    shift : int
        Number of pixels to wrap the image
    axis : int
        Axis for wrapping

    Returns
    -------
    numpy array
        A shifted image

    """
    wrapped_images = []
    for im in images:
        wrapped_images += [
            wrap_image(im, shift=shift, axis=axis)]
    return np.asarray(wrapped_images)


# Test the impact of vertical wraps
wraps = np.arange(1, 6)
mlp_perf, conv_perf = [], []
for w in wraps:
    wrapped_x_test = apply_wraps(
        images=normalized_x_test,
        shift=w,
        axis=0)
    mlp_perf += [mlp_model.evaluate_generator(
        datagen.flow(
            ,
            ,
            batch_size=batch_size),
        steps=len(x_test))[1]]
    conv_perf += [conv_model.evaluate_generator(
        datagen.flow(
            ,
            ,
            batch_size=batch_size),
        steps=len(x_test))[1]]

f, ax = utilities_module_3.plot_performance(
    ,
    colors=['b', 'r'],
    label='MLP')
utilities_module_3.plot_performance(
    ,
    colors=['g', 'k'],
    label='Convolution',
    f=f,
    ax=ax)
plt.savefig('module_3_assigment_1_Q3_figure.png')
plt.show()
plt.close(f)


# Test the impact of horizontal wraps
wraps = np.arange(1, 6)
mlp_perf, conv_perf = [], []
for w in wraps:
    wrapped_x_test = apply_wraps(
        images=normalized_x_test,
        shift=w,
        axis=1)
    mlp_perf += [mlp_model.evaluate_generator(
        datagen.flow(
            ,
            ,
            batch_size=batch_size),
        steps=len(x_test))[1]]
    conv_perf += [conv_model.evaluate_generator(
        datagen.flow(
            ,
            ,
            batch_size=batch_size),
        steps=len(x_test))[1]]

f, ax = utilities_module_3.plot_performance(
    mlp_perf,
    colors=['b', 'r'],
    desc='Accuracy',
    label='MLP')
utilities_module_3.plot_performance(
    conv_perf,
    colors=['g', 'k'],
    label='Convolution',
    desc='Accuracy',
    f=f,
    ax=ax)
plt.savefig('module_3_assigment_1_Q4_figure.png')
plt.show()
plt.close(f)
