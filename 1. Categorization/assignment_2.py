"""
## Module 1 assignment 2: Cats vs. dogs


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 2 and 3 (3 total).
    These questions are denoted Q1, etc. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_1_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_1_assignment_2.zip

## Description

In the previous lecture, you learned about how the brain categorizes images.
In Freedman et al (2001), it was found that some neurons in the prefrontal
cortex (PFC) learn to categorize between cats and dogs. This was tested by
showing participants stimuli that morphed along a continuum of cat to dog
prototypes (from 100% cat to 100% dog in 20% steps). Given a fixed perceptual
dissimilarity between two images (say 20% morph), neural responses were much
more similar for images that belonged to the same category (say 80% cat vs 60%
cat) compared to images that crossed the categorical boundary (say 60% cat to
40% cat).

In this assignment, you will show that a computational model of categorization
behaves similarly to prefrontal cortex. You are given an artificial neuron
that has already learned to discriminate between cats and dogs. You do not
have to worry about how the neuron has learned for now (this will be done in
the next lecture). You are given a set of weights and you will use these to
classify images as either cats or dogs.

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 45-60 mins.

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
import re
import numpy as np
from scipy import misc
from glob import glob
from utils import py_utils, data_utils, plot_utils, utilities_module_1
from matplotlib import pyplot as plt


"""
## Find the images.
"""
im_dir = os.path.join('data')
im_ext = '.bmp'
files = np.asarray(glob(os.path.join(im_dir, '*%s' % im_ext)))

"""
## Load cats and dogs images.
"""
ims = []
for f in files:
    ims += [misc.imread(f)]
assert len(ims), 'Could not find any images. Did you download them? Are they in the folder: %s' % im_dir
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
## Visualize the images.
"""
cat_ims = [x for x, label in zip(ims, labels) if label == 0]
dog_ims = [x for x, label in zip(ims, labels) if label == 1]
py_utils.plot_im_grd(cat_ims, title='Cat images')
py_utils.plot_im_grd(dog_ims, title='Dog images')

"""
## Reshape ims into a (example) x (pixel) matrix.
"""
ims = ims.reshape(num_ims, height * width * channels)

"""
## Grab images that illustrate a dog-cat gradient and look at classifier
performance on them.
"""
sorted_dc_ims, sorted_dc = utilities_module_1.get_dc_ims(files)

# Take a look at the images and prepare them for the model
py_utils.plot_im_grd(sorted_dc_ims, title='Dog-cat image gradient')
res_dc_ims = sorted_dc_ims.reshape(len(sorted_dc), -1)


"""
We trained the model on normalized versions of the images:
At each location, a pixel intensity is transformed into a z-score according
to the mean and standard deviation of pixels intensities for that location
accross all images. For the expert, this is essentially assuming that pixel
intensities at each location are normally distributed and each intensity is
rescaled according to the amount it deviates (in number of standard deviations)
from the mean across all images.

A z-score of 0 means that the pixel intensity is equal to the average
A z-score of 1.7 means that the pixel intensity falls 1.7 standard deviations
above the average
A z-score of -2 means that the pixel intensity falls 2 standard deviations
below the average

Normalizing the training data usually helps the classifier learns more easily.
With unnormalized data, a single step in a gradient descent (or similar)
algorithm might change one of the weights by much more (proportionally) than
one of the other weights.

For the trained model to be applicable to new data, we need to normalize the
new data as well (according to the same distribution).
"""
train_mu = np.load(os.path.join('models', 'train_mu.npy'))
train_sd = np.load(os.path.join('models', 'train_sd.npy'))

norm_dc_ims, _, _ = data_utils.normalize(
    res_dc_ims,
    method='zscore',
    mu=train_mu,
    sd=train_sd)

"""
###########################
1. EDIT HERE!
###########################
Your first task is to make predictions about 10 images that lie along a
continuum from most dog-like to most cat-like. If the model is working
properly, it should confidently classify dog images as dogs and cat
images as cats.

The data is stored in a variable called norm_dc_ims. See the top of this script
for more information on how this variable was created.

The model learned weights to predict the probability that an input image
is a dog. The weights are related to the prediction by the following formula:
f(x) = sigmoid(wx+b) = 1 / (1 + exp(-wx-b))
where f(x) is the the predicted probability that image vector x is a dog.
b is the bias term (equivalent to an additional weight learned during training)
W is a vector of weights (each representing the extent to which the
corresponding input variable influences the prediction), also learned during
training.

Here wx is a matrix multiplication of a [1 x pixels] vector and an [pixels x 1]
vector (also known as a dot product).

The expression (wx+b) will be a real number that gets larger when x lies
more in the direction of the weight vector w, and smaller when
x lies more in the opposite direction of w. (wx+b) specifies a (hyper)plane in
space: this is our decision boundary. Points on one side of the plane are
classified as dogs, and points on the other side are classified as cats.

The sigmoid function takes this (potentially quite large or small) number
and maps it to a number between 0 and 1 that can be interpreted as a
probability:
"""
w, b = np.load(
    os.path.join(
        'models',
        'weights.npy'))

"""
Now we have everything we need to make predictions about the data in
norm_dc_ims.

Obtain a vector in which the ith entry contains the prediction our model
makes about the ith image in norm_dc_ims (for all i).

Hint: you can do this without a for loop! think about the shape of your data
and weights. your data (norm_dc_ims) is [examples x pixels], W is
[pixels x 1], and b is [1 x 1].

Q1: In two or three sentences explain what the plot you produce on line 204
shows. Include the figure in your response (this will be saved as
"module_1_assigment_2_Q1_figure.png")

Estimated time: 10-20 minutes

"""
# Your code here:
dc_probabilities =

utilities_module_1.plot_dc_gradient(  # Plot scores for these images
    res_dc_ims,
    sorted_dc,
    dc_probabilities)

"""
###########################
2. EDIT HERE!
###########################
Let's see how well the model generalizes to data it hasn't seen before. In
machine learning we refer to data held-out of model training as "test data".

Report the test accuracy of the classifer. Interpret any probability greater
than 0.5 as a prediction of a dog, and any probability 0.5 or smaller as a
prediction of a cat.

The test accuracy is the percentage of test examples for which our
prediction matches the label.


Estimated time: 5-10 minutes.
"""
test_indices = np.load(
    os.path.join(
        'models',
        'test_indices.npy'))
testing_ims = ims[test_indices]
testing_labels = np.array(labels[test_indices]).reshape(-1, 1)

norm_test_ims, _, _ = data_utils.normalize(
    testing_ims,
    method='zscore',
    mu=train_mu,
    sd=train_sd)

# Your code here:
test_preds = 
# Note: you may get an overflow warning about this line. This is fine,
# it just means the classifier is very confident about its prediction for
# some image.
test_accuracy = 
print('Test Accuracy: %s' % test_accuracy)

"""
###########################
3. EDIT HERE!
###########################
## Plot the least and most cat/dog images.

Now let's make a prediction about every image (including the training data)
and explore the data through the lens of our classification unit.

(By now you already know how to make predictions. The challenge is to use
the information stored in the variables on lines 222-225 to fill in the
variables on lines 233-236.)

Q2: What kinds of features denote the "doggiest dog" versus the "cattiest cat"?
How do these compare to the "cattiest dog" and the "doggiest cat"?

Q3: Compare and contrast the classifer's dogginess score across these images
to recordings in primate cortex from Freedman et al (2001) discussed in
lecture. Include the figure you produce on line 339 (the classifier's scores
corresponding to the continuum of dogs to cats).

Estimated time: 10-20 minutes
"""

norm_ims, _, _ = data_utils.normalize(
    ims,
    method='zscore',
    mu=train_mu,
    sd=train_sd)

# Your code here:
all_preds = 
# Note: you may get an overflow warning about this line. This is fine,
# it just means the classifier is very confident about its prediction
# for some image(s).

cat_ims = ims[labels == 0]
dog_ims = ims[labels == 1]
cat_preds = all_preds[labels == 0]  # predictions made on cat images
dog_preds = all_preds[labels == 1]  # predictions made on dog images
"""
Fill these variables with the corresponding element of cat_ims or dog_ims
By "doggiest" we mean most doglike according to our classifer. By "dog",
we mean that the image was labeled a dog. Remember that a prediction closer to
1 corresponds to a "doggier" image.
"""
# Your code here
cattiest_cat = 
doggiest_cat = 
cattiest_dog = 
doggiest_dog = 

f, ax = plt.subplots(1, 4)
ax[0].imshow(cattiest_cat.reshape(height, width, channels), cmap='Greys')
ax[0].set_title('Cattiest cat')
ax[1].imshow(doggiest_cat.reshape(height, width, channels), cmap='Greys')
ax[1].set_title('Doggiest cat')
ax[2].imshow(cattiest_dog.reshape(height, width, channels), cmap='Greys')
ax[2].set_title('Doggiest dog')
ax[3].imshow(doggiest_dog.reshape(height, width, channels), cmap='Greys')
ax[3].set_title('Cattiest dog')
plt.show()
plt.close(f)

"""
Finally, we will examine how the classification of all of our images compares
to the information we have about where the images lie along their respective
continuums:
"""

buckets = utilities_module_1.bucket_ims(files)
bucket_sums = np.zeros([1, 14])
bucket_totals = np.zeros([1, 14])

for i, pred in enumerate(all_preds):
    buck = buckets[files[i]]
    if buck > 0:
        bucket_sums[0, buck - 1] += pred
        bucket_totals[0, buck - 1] += 1

bucket_averages = bucket_sums / bucket_totals
for i, avg in enumerate(bucket_averages[0]):
    plt.plot(i + 1, avg, 'g*', linewidth=3)

plt.xticks(range(1, 15))
plt.title('Average classifier confidence across all images')
plt.ylabel('Confidence in dog')
plt.xlabel('Images ranked from most dog (1) to most cat (14)')
plt.savefig('module_1_assigment_2_Q3_figure.png')
plt.show()
