"""
## Module 3 assignment 2: System identification with brains and machines


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 1, 2, 3, and 4. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_3_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_3_assignment_2.zip

## Description

System identification is a powerful technique for understanding the brain, and
specifically neurons in the visual system. The response profile of parts of
visual cortex is the basis for the architectures of deep neural networks, and
the current state of has its foundations in these findings.

One approach for system identification is to estimate the shape of the filter
used by neurons to encode the visual scene. For instance, some neurons may be
sensitive to vertically oriented edges and others to horizontally oriented
edges. It turns out that these profiles can be inferred by recording responses
to a wide array of stimuli, and using those responses to derive a reweighted
combination of the stimuli. As you learned in lecture, approaches of this ilk
are called Spike Triggered Average (STA) or Spike Triggered Covariance (STC)
analyses, where the number of spikes a stimulus elicits from a neuron is used
to derive its response profile.

In computer vision, we can read continous-valued responses from
neurons/computational units in our network to carry out a similar analysis.

In this assignment you will learn about STA and STC, and how to perform these
analyses on a multilayer neural network. You will also estimate parameters to
describe the response properties of neurons in your network, and compare those
parameters to ones similarly estimated from STC-derived profiles of primate
early visual cortex (V1).

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 2 hours.

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
from utils import utilities_module_3
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

"""
###########################
1. EDIT HERE!
###########################

Create a model and train it on CIFAR images, then do both STA and STC on it.

You will use white noise to do these analyses.

Q1. Create a figure of STA on your model. Plot a separate figure for STC. What
does each reveal about the response profile of your model? Do the two methods
reveal different information about the response profile? How do the
techniques when you change N, measuring respoinses to fewer or more white noise
examples?

Estimated time: 45 minutes.
"""

"""
## Load and plot the filters.
"""

norm_filters, filters, biases = utilities_module_3.load_normalized_filters(
    path=os.path.join(
        'data',
        'alexnet.npy'))
num_filts, filt_height, filt_width, filt_channels = filters.shape

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

"""
## Normalize your images.
"""
normalized_x_train = utilities_module_3.normalize_ims()
normalized_x_test = utilities_module_3.normalize_ims()
number, height, width, channels = normalized_x_train.shape

# One-hot encode labels and create a data generator
ey_train =
ey_test =
datagen = ImageDataGenerator(  # GRADE
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False)
datagen.fit()

"""
## Prepare your model.
"""
epochs =
output_dim =
lr = 3e-4
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)

"""
## Create your conv model.
"""
conv_model = Sequential()
conv_model.add(
    Conv2D(
        num_filts,
        (filt_height, filt_width),
        activation='relu',
        input_shape=(height, width, channels)))
conv_model.add(
    Flatten())
conv_model.add(
    Dense(
        output_dim,
        activation='softmax'))
conv_model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])
conv_model_weights = conv_model.get_weights()
conv_model_weights[0] = filters.transpose(1, 2, 3, 0)
conv_model_weights[1] = biases
conv_model.set_weights(conv_model_weights)

"""
## Train your conv model.
"""
conv_output = conv_model.fit_generator(
    datagen.flow(  # Grade
        normalized_x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        normalized_x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
conv_train_loss = conv_output.history['loss']
conv_test_loss = conv_output.history['val_loss']

# White noise STA
N = 10000  # How many white noise exemplars?
white_noise = np.random.uniform(
    low=normalized_x_train.min(),
    high=normalized_x_train.max(),
    size=[N] + list(normalized_x_train.shape[1:]))
reshaped_white_noise = utilities_module_3.unravel(white_noise)

# Calculate each convolutional filter's activity elicited by the white noise.
white_noise_response = conv_model.predict(white_noise)

"""
## Calculate the response triggered average (STA) for the model.

You can calculate this as:

t = x_T

STA = t . z

where STA is the dot product of the transposed and reshaped white noise
matrix t with the model's response to each white noise exemplar z.

"""
STA = 
nSTA = utilities_module_3.normalize_to_zero_one(STA.transpose(), axis=1)
reshaped_nSTA = 
utilities_module_3.plot_mosaic(
    reshaped_nSTA,
    title='STA',
    rows=2,
    columns=2)

"""
## Calculate the response triggered covariance (STC) for the model.

You can calculate this as:

t = x_T

STA = (t . t_T)^-1 . (t . z)

where STA is the least squares solution fitting t with z. You can solve
this by hand (not recommended) or by using numpy's built-in solver.
"""
# Now try estimating the response-triggered covarinace
STC = np.linalg.lstsq(reshaped_white_noise, white_noise_response)
nSTC = utilities_module_3.normalize_to_zero_one(STC[0].transpose(), axis=1)
reshaped_nSTC = nSTC.reshape(nSTC.shape[0], height, width, channels)
utilities_module_3.plot_mosaic(
    reshaped_nSTC,
    title='STC',
    rows=2,
    columns=2)


"""
###########################
2. EDIT HERE!
###########################

In the above exercise you used white noise to perform STA. What
happens when you use natural images instead?

Q2. Plot STA derived from images you used to train your model. How does
this compare to the white noise examples above? Why do you think there are
differences? Include a plot of the filters in your response


Estimated time: 15 minutes.
"""

image_response = conv_model.predict(normalized_x_train)

# Reshape the images for STA
n_images = utilities_module_3.unravel(normalized_x_train)

# Calculate the response triggered average (STA) for the ZCA images
image_STA = np.dot(n_images.transpose(), image_response)
image_STA = utilities_module_3.normalize_to_zero_one(
    image_STA.transpose(), axis=1)
reshaped_STA = image_STA.reshape(image_STA.shape[0], height, width, channels)
utilities_module_3.plot_mosaic(
    reshaped_STA,
    title='STA',
    rows=2,
    columns=2)


"""
###########################
3. EDIT HERE!
###########################

One way of getting reliable STA from natural images is by first "whitening"
them to decorrelate neighboring pixels. Here we use an approach called
ZCA whitening to do that.

Q3. Run STA on the zca-whitened images. How do the recovered filters look, and
how do they compare to the ones you recovered from unwhitened images? Include
a plot of the filters in your response

Estimated time: 15 minutes.
"""


def zca_whiten(x):
    """ZCA whiten the image tensor x.

    Zero-phase component analysis (Bell and Sejnowski, 1996; ZCA) whitening
    is a method for decorrelating pixels in images. This is an important
    preprocessing step when using natural images to estimate STA or STC. The
    local correlations present in natural images are incompatible with
    STA and STC, necessitating their removal.

    Parameters
    ----------
    x : float
        [n, m] numpy image matrix

    Returns
    -------
    numpy array
        The ZCA whitening matrix
    numpy array
        A tensor of normalized activities

    """
    # Normalize x if necessary
    mu = x.mean(0)
    std = x.std(0)
    x = (x - mu) / std

    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(x, rowvar=True)

    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U

    # Whitening constant: prevents division by zero
    epsilon = 1e-5

    # ZCA Whitening matrix: U * Lambda * U'
    zca_matrix = np.dot(
        U, np.dot(
            np.diag(1.0 / np.sqrt(S + epsilon)),
            U.T))
    zca_transform = np.dot(zca_matrix, x)
    return zca_matrix, zca_transform, mu, std


# Reshape the images
flat_x_train =

# Whiten the images
zca_matrix, zca_x, mu, std =

# Reshape the flattened image matrix into a tensor of images
zca_images =

# View an image pre- and post-whitening
pre_zca = x_train[0]
post_zca = zca_images[0]
f = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(pre_zca)
plt.title('Pre-zca')
plt.subplot(1, 2, 2)
plt.imshow(utilities_module_3.normalize_to_zero_one(post_zca))
plt.title('Post-zca')
plt.show()


# Normalize ZCA images before passing through the model
nzca_images = zca_images / 255.
nzca_images = []
for im in zca_images:
    im /= 255.
    im = im[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    im -= im.mean()
    nzca_images += [im]
nzca_images = np.asarray(nzca_images)

# Calculate each convolutional filters activity elicited by ZCA images.
zca_response =

# Reshape the ZCA images for STA/STC
nzca_images =

# Calculate the response triggered average (STA) for the ZCA images
zca_STA =
zca_normalized_STA =
reshaped_zca_normalized_STA =
utilities_module_3.plot_mosaic(
    reshaped_zca_normalized_STA,
    title='STA',
    rows=2,
    columns=2)

"""
###########################
4. EDIT HERE!
###########################

Can we compare brains and machines? The response profile of early visual cortex
(V1) neurons is typically modeled with a gabor function. Research such as you
learned about in lecture (Ringach, 2001) has described a linear relationship
between the height and widths of gabors fit to V1 neuron response profiles.

Does this relationship hold for your convolutional filters, which were trained
on natural images?

Here you will fit gabors to STA on zca-normalized images, and
compare the recovered height (y-axis sigma) and width (x-axis sigma) to the
same parameters inferred from V1 neurons.

Q4. Plot gabor parameters from your model against V1. Describe the results and
what they suggest about vision and learning.

Estimated time: 15 minutes.
"""


# Fit gabors to the filters. This returns two sigmas (widths): one describing
# the gabor's x-axis and the other describing its y-axis
STA_fits = np.asarray(
    [utilities_module_3.gabor_fit(
        reshaped_zca_normalized_STA[idx].mean(-1))[2:4] for idx in range(
        reshaped_zca_normalized_STA.shape[0])])

# Read the CSV containing gabor fits to cells in primate V1
neural_fits = np.genfromtxt(
    os.path.join(
        'data',
        'ringach_2001.csv'),
    delimiter=',')

# Normalize the STA_fits and neural_fits so that we can compare them
STA_fits = (STA_fits - STA_fits.mean(0)) / STA_fits.std(0)
neural_fits = (neural_fits - neural_fits.mean(0)) / neural_fits.std(0)

f = plt.figure()
plt.plot(
    ,
    ,
    'b.',
    label='Primate V1')
plt.plot(
    ,
    ,
    'rs',
    label='Convolutional filters (Alexnet)')
plt.xlabel('X sigma (normalized units)')
plt.ylabel('Y sigma (normalized units)')
plt.legend()
plt.title(
    'Linear relationship in shape of filters in both primate V1 and machines')
plt.show()
