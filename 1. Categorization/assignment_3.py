"""
## Module 1 assignment 3: Gradient Descent


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 1, 3, 4, and 7 (4 total).
    These questions are denoted Q1, etc. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_1_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_1_assignment_3.zip


## Description

In this assigment, you will implement the gradient descent algorithm
to train a logistic regression classifier on the cat/dog dataset.

Fill in missing code to complete the assignment. These portions of the
code are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 1 - 2 hours.

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
from scipy.stats import binom_test
from matplotlib import pyplot as plt


"""
## Find the images
"""
im_dir = os.path.join('data')
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
## Reshape ims into a (example) x (pixel) matrix.
"""
ims = ims.reshape(num_ims, height * width * channels)

"""
## Split files into train and test sets.
"""
training_split = 0.9  # Hold out this proportion for training
dog_labels = np.where(labels == 1)[0]
cat_labels = np.where(labels == 2)[0]
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

norm_train_ims, train_mu, train_sd = data_utils.normalize(
    training_ims,
    method='zscore')


def sigmoid(x):
    """Sigmoidal function.

    Parameters
    ----------
    x : float
        A float of any shape.


    Returns
    -------
    float
        x after an element-wise sigmoid transformation.
    """
    return 1 / (1 + np.exp(-x))


"""
We are ready to train a logistic regression classifier for
our dataset using gradient descent.

First, we initialize the variables we are optimizing (W and b)
to random numbers. We also define our learning rate (lr) and the number
of iterations (epochs) of gradient descent that we will run.
"""

# Model data.
input_dims = norm_train_ims.shape[-1]  # Dimensions of your images x.
output_dims = 1  # Dimensions of your labels y.
m = norm_train_ims.shape[0]  # Number of examples

# Model training variables.
epochs = 100  # Number of epochs of running through the training images.
eps = 1e-12  # A constant that protects from overflow.
lr = .1  # Learning rate.

# Initialize learned weights.
mu_w = 0.  # Mean of randomly initialized weights.
sd_w = .1  # Standard deviation of randomly initialized weights.
size_w = [input_dims, output_dims]  # Size of learned weight matrix.
mu_b = 0.  # Mean of randomly initialized bias.
sd_b = .1  # Standard deviation of randomly initialized bias.
size_b = [output_dims]  # Size of learned bias.

# Create model variables.
x = norm_train_ims  # Rename variables for consistency with formulae.
y = np.expand_dims(training_labels, axis=-1)  # Turn Y into a matrix.
w = np.random.normal(loc=mu_w, scale=sd_w, size=size_w)
b = np.random.normal(loc=mu_b, scale=sd_b, size=size_b)

# Run your model for a number of epochs.
for i in range(epochs):
    """
    ###########################
    1. EDIT HERE!
    ###########################
    Below is the first step in the forward pass, where we find
    our predictions for every datapoint, and store them in the vector p

    Remember that a logistic regression classifier classifies based
    on f(x) = sigmoid(z), where z = xw + b, and x is a single image.
    Here we denote: y_hat = sigmoid(z). In words, p is a sigmoid
    transformed projection of the image x on the decision boundary w.

    Hint: you can calculate z and y_hat in a single line each.
    Pay attention to the shape of these variables:
    x is [m, input_dims], w is [input_dims, output_dims], b is
    [output_dims]. We want z and p to both be [m, output_dims].

    The sigmoid function we wrote for you is vectorized, so you
    can pass it a vector [z_1, z_2, z_3] (where z_1 refers to]
    the first dimension of z and so on). The function applies an
    element-wise sigmoid to z: [sigmoid(z_1), sigmoid(z_2), sigmoid(z_3)].

    It is unnecessary to use a for loop to add a constant to
    every value in a vector: np.array([x_1, x_2, x_3]) + 4 returns the
    array([x_1 + 4, x_2 + 4, x_3 + 4]).

    Q1: In two or three sentences, describe the operations of your forward pass
    and what they accomplish.

    Estimated time: 5 minutes.
    """
    # Forward pass
    z = 
    y_hat = 

    """
    We use a loss function to measure how accurate our predictions
    are at the current iteration. This function gets smaller when a
    matches Y, so we can descend its gradient to find a set of
    weights (and bias) that gives us better predictions.

    We can calculate a prediction for a single example as:
    z = xw + b
    y_hat = sigmoid(z)

        NOTE: In lectures you learned an objective function called mean squared
        error in the form of:

        e(w) = mean((y - yhat) ** 2)

        In practice, categorization problems benefit from an alternative error
        function called crossentropy. Below we define this loss for a binary
        class problem (this is more generally known as binary cross entropy).

        Also, in lecture we call this an objective/cost function. It is more
        generally referred to as a "Loss" function in deep learning. We use
        therefore will use the term loss hereafter.

    We can write the loss e for a single example as:
    e = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))

    So when y is 1, only the first term matters, and we minimize L by
    maximizing a. When y is 0, only the second term matters, and we
    minimize L by minimizing a.

    The eps terms are to avoid rounding errors that might otherwise
    force us to take the log of 0
    """

    e = -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    """
    ###########################
    2. EDIT HERE!
    ###########################
    Next, we need to compute the gradient of our loss function with
    respect to our variables W and b

    We will do this math in terms of a single example x and its label y.
    Once again, you will be able to do this code without for loops by
    taking advantage of matrix multiplication.

    e = - (y * log(y_hat) + (1 - y) * log (1 - y_hat))
    y_hat = sigmoid(z) = 1 / (1 + exp(-z))
    z = xw + b

    We want de/dw and de/db. We will use the chain rule:
    de/dw = de/dy_hat * dy_hat/dz * dz/dw
    de/db = de/dy_hat * dy_hat/dz * dz/db

    de/dw = -(y / y_hat) + (1 - y) / (1 - y_hat)
    dy_hat/dz = exp(z) / (exp(z) + 1)^2
    de/dz = de/dy_hat * dy_hat/dz = y_hat - y

    (you may enjoy seeing by hand how the previous two lines multiply
    out to something as simple as a-y).

    dz/dw = [x_0, x_1, ...]
    dz/db = 1

    in summary: dz = y_hat - y
    de/dw = [x_0 dz, x_1 dz, ...]
    de/db = dz

    This is for a single training example. In your code, dw and db should
    store the accumulation of these values across all m examples. So dW should
    look like:
    [
        (x[0, 0] * dz[0] + x[0, 1] * dz[0] + ...),
        (x[1, 0] * dz[1] + x[1, 1] * dz[1] + ...),
        ...
    ]
    Where x[i, j] is the value of the jth dimension of x for example i, and
    dz[i] is the value of dz for training example i.

    Fill out the following lines. We omit the de/ in the variable name,
    so dz represents de/dz

    Hint: each of these variables can be assigned in one line.
    Once again, it will be helpful to think about the shapes of each variable.

    y, y_hat, z, dz: [m, output_dims]
    w, dw: [input_dims, output_dims]
    x: [m, input_dims]

    Estimated time: 15-30 minutes.
    """
    # Calculate updates
    dz = 
    dw = 
    db = 

    """
    ###########################
    3. EDIT HERE!
    ###########################
    Finally, we want to update W and b in the opposite direction of their
    gradient. We use a learning rate lr to prevent us from taking huge steps.

    For example, let's say we are trying to find the minimum of the
    graph y = x^2, and our current value of x is 3. The value of the gradient
    at x=3 is dy/dx = 2x = 6. If our learning rate was lr = .1, we would set
    our new value of x to be 3 - .1(6) = 2.4

    Q2: In two or three sentences, describe the operations of your
    backward pass and what they accomplish.

    Estimated time: 5 minutes.
    """
    # Apply updates to weights
    w -= 
    b -= 

    # Print the loss
    print('Training epoch: %s | Loss: %s' % (i, e))

"""
Now that we've trained our classifier, let's visualize it!
W should have height x width x channels entries. We can reshape
W into a matrix such that individual weights appear in the position
of the pixel they correspond to. This lets us visualize for each
pixel the extent to which it influences our classifier, and in which direction.

You should see that the model has learned a representation of
what makes a dog a dog and a cat a cat. The values of your
weights indicate how a pixel in every location is interpreted:
white corresponds to positive weights and an increased probability
of dog classification, whereas black corresponds to negative
weights and an increased probability of cat classification.
"""
rw = w.reshape(height, width, channels)  # Reshape weight vector to an image.
nw = (rw - rw.min()) / (rw.max() - rw.min())  # Normalize to [0, 1].
f = plt.figure()
plt.imshow(nw, cmap='Greys')
plt.title('Your Weights')
plt.show()
plt.close(f)

"""
###########################
4. EDIT HERE!
###########################

Now let's compare the model you just built to the model we gave you in
assigment 2.

We can plot your model on the left, and our model on the right.
Do you notice any differences?

Q3: Describe the kinds of visual features the classifier uses to discriminate
between dogs and cats. Prior to normalization, pixels take on values that are
positive and negative (after normalization these values approach 0 and 1,
respectively). What do they tell you about how your model classifies dogs
versus cats? Include your figure in your response.

Estimated time: 5 minutes.

"""
teacher_w, teacher_b = np.load(os.path.join('models', 'weights.npy'))
rteacher_w = teacher_w.reshape(height, width, channels)  # As in line 290
normteacher_w = (
    rteacher_w - rteacher_w.min()) / (rteacher_w.max() - rteacher_w.min())

f = plt.figure()
plt.subplot(121)
plt.imshow(nw, cmap='Greys')
plt.title('Your Weights')
plt.subplot(122)
plt.imshow(normteacher_w, cmap='Greys')
plt.title('Our Weights')
plt.savefig('module_1_assigment_3_Q3_figure.png')
plt.show()
plt.close(f)

"""
###########################
5. EDIT HERE!
###########################
How does your model compare to ours in terms of test accuracy?
Fill out the following function to compute the accuracy of a model
given its weights, bias, test data, and test labels.

Estimated time: 2 minutes
(you can essentially copy this code from m1_a2)
"""


def compute_test_accuracy(w, b, test_x, test_y):
    """
    Calculate accuracy on test y with model (w, b).

    With m test examples of n-dimensional data:
    w: [n, 1] numpy array of weights
    b: [1] numpy array representing the bias
    test_x: [m, n] array of normalized test data
    test_y: [m, 1] array of test labels

    Parameters
    ----------
    w : float
        [n, 1] vector of weights
    b : float
        [1] bias
    test_x : float
        [m, n] test data normalized to 0 mean and unit variance
    test_y : int
        [m, 1] labels

    Returns
    -------
    float
        accuracy of predictions versus test_y
    """
    y_hat = sigmoid(np.dot(test_x, w) + b)
    accuracy = np.mean(test_y == np.round(y_hat))
    return accuracy


# Load indices for test data and means/sds from training data.
test_indices = np.load(os.path.join('models', 'test_indices.npy'))
train_mu = np.load(os.path.join('models', 'train_mu.npy'))
train_sd = np.load(os.path.join('models', 'train_sd.npy'))
testing_ims = ims[test_indices]
testing_labels = np.array(labels[test_indices]).reshape(-1, 1)


# Normalize test images with train means/sds.
norm_test_ims, _, _ = data_utils.normalize(
    testing_ims,
    method='zscore',
    mu=train_mu,
    sd=train_sd)

# Calculate the supplied model's accuracy on test images.
test_accuracy = compute_test_accuracy(
    w=teacher_w,
    b=teacher_b,
    test_x=norm_test_ims,
    test_y=testing_labels)
print('Test Accuracy (our weights): %s' % test_accuracy)

# Calculate your new model's accuracy on test images.
test_accuracy = compute_test_accuracy(
    w=w,
    b=b,
    test_x=norm_test_ims,
    test_y=testing_labels)
print('Test Accuracy (your weights): %s' % test_accuracy)

"""
###########################
6. EDIT HERE!
###########################
How can we interpret these accuracy numbers?
One sanity check is to obtain a p-value for our result.
We ask the question "How often would someone who is flipping
coins randomly to determine if the images are cats or dogs
score as well or better than us?"

In statistical language, we have a null hypothesis that our
classifier has a 50 percent chance of correctly classifying
an image.

Our alternative hypothesis is that we have a greater than 50
percent chance of correctly classifying an image.

We often interpret a p-value greater than .05 as meaning that
we have insufficient evidence to reject the null hypothesis
(in our case, this would mean we should not be confident that
we are doing better than chance).

Estimated time: 5 minutes.
"""
null_p =  # Probability of correct classification under the null hypothesis.
attempts = 
successes = 

p = binom_test(successes, attempts, null_p, alternative='greater')

print('Your p-value is: %s' % p)

"""
You may have noticed that it is difficult to obtain a low
p-value with such a small number of test images. If you'd like,
try re-running the script using different values for the
training_split variable.
"""


"""
###########################
7. EDIT HERE!
###########################
You have trained neural models to distinguish between cats and
dogs, paralleling the result from Freedman et al., 2001 that
was discussed in lecture.

What if you wanted a model for finer-grained classification?
As you saw in assignment two, the dog and cat categories
contain distinct cat and dog "species". Your next task is to
train a model to categorize between these different kinds of
cats and dogs.

To do this you will turn the training routine you developed
above into a function that can handle a multiclass input.
Note that the error/cost function has changed, but otherwise
your backwards pass and updates are the same. Fill in the
"train_model" function below to complete this part of the
assignment.

Important: the error function implemented below is called categorical
crossentropy (CCE). The binary crossentropy (BCE) you used to train your
previous model is a special case of CCE:

CCE: e = np.mean(-np.sum(y * np.log(y_hat), axis=1))

BCE: e = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

Because of the different form of the CCE loss, your labels must be
formatted differently. Previously, your labels were a binary vector.
They must now be reformatted into a "one-hot" encoding. This means
that they will be transformed from a vector of integers in the range
of [1, 6] to a 6-dimensional array. Each row is made up of zeros
except for the column corresponding to the category label. This means,
for example, that the label [5] would be transformed to the vector:
[0 0 0 0 1 0].

We use vectorized numpy operations to implement this one-hot encoding:
np.eye(mc_train_labels.max() + 1)[mc_train_labels]

This means you preallocate an identity matrix (np.eye) that has the same
number of dimensions as there are unique labels in thos dataset. Each entry
in mc_train_labels draws a row from the identity matrix, which provides a
one-hot encoding.

Q4: Compare and contrast the softmax nonlinearity to the sigmoid
you used early. When are they the same and when are they different?
Also, describe the kinds of visual features your model learns to
classify multiple categories at once. Include your figure with
your response.

Estimated time: 20 minutes.

"""


def train_model(
        x,
        y,
        epochs=200,
        lr=.001,
        eps=1e-12,
        mu=0.,
        sd=.1):
    """Train a multiclass classifier.

    Parameters
    ----------
    x : float
        [m, n] train data normalized to 0 mean and unit variance
    y : int
        [m, 1] labels
    epochs : int
        Number of epochs of training
    lr : float
        Learning rate for gradient descent
    eps : float
        Constant to avoid divide by 0 errors
    mu : float
        Initialization mean for weights
    sd : float
        Initialization standard deviation for weights

    Returns
    -------
    float
        weights for the model
    float
        bias for the model
    """
    def softmax(x, eps=1e-12, axis=1):
        """
        Non-linearity for multiclass classification.
        A generalization of the sigmoid to multiple
        classes.

        Parameters
        ----------
        x : float
            A matrix of logits
        eps: float
            Constant to avoid divide by 0 errors
        axis: int
            Axis for applying softmax normalization

        Returns
        -------
        float
            x following softmax normalization
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + eps)

    input_dims = x.shape[-1]  # Number of input dimensions
    output_dims = y.shape[-1]  # Number of target dimensions
    size_w = [input_dims, output_dims]  # Size of learned weight matrix.
    size_b = [output_dims]  # Size of learned bias.

    # Initialize learned weights.
    w = np.random.normal(loc=mu, scale=sd, size=size_w)
    b = np.random.normal(loc=mu, scale=sd, size=size_b)

    # Run your model for a number of epochs.
    for i in range(epochs):

        # Forward pass.
        z =
        y_hat =

        # Ensure y_hat is in [0, 1].
        clipped_y_hat = np.clip(y_hat, eps, 1 - eps)

        # Use a categorical loss (cross-entropy).
        e = np.mean(-np.sum(y * np.log(clipped_y_hat), axis=1))

        # Backward pass updates
        dz =
        dw =
        db =
        w -=
        b -=

        # Print the loss
        print('Training epoch: %s | Loss: %s' % (i, e))
    return w, b


# Prepare data for multiclass classification
ims = []
for f in files:
    ims += [misc.imread(f)]
assert len(ims), 'Could not find any images. Did you download them? Are they in the folder: %s' % im_dir
ims = np.asarray(ims)
md = utilities_module_1.derive_mc_labels(
    files=files,
    images=ims)
mc_train_labels = md['train_labels']
mc_train_images = md['train_images']
mc_test_labels = md['test_labels']
mc_test_images = md['test_images']

# Convert labels to multiclass "one-hot" encodings
mc_train_labels = np.eye(mc_train_labels.max() + 1)[mc_train_labels]
mc_test_labels = np.eye(mc_test_labels.max() + 1)[mc_test_labels]

# Normalize the images
num_ims, height, width, channels = mc_train_images.shape
mc_train_images = mc_train_images.reshape(num_ims, height * width * channels)
mc_norm_train_ims, train_mu, train_sd = data_utils.normalize(
    mc_train_images,
    method='zscore')
num_ims, height, width, channels = mc_test_images.shape
mc_test_images = mc_test_images.reshape(num_ims, height * width * channels)
mc_norm_test_ims, _, _ = data_utils.normalize(
    mc_test_images,
    method='zscore',
    mu=train_mu,
    sd=train_sd)

# Run multiclass classification
mc_weights, mc_b = train_model(
    x=mc_norm_train_ims,
    y=mc_train_labels)


# Visualize each of the category weights
def normalize_weights(weights, height, width, channels):
    """Reshape and normalize weights.

    Parameters
    ----------
    weights : float
        [m, n] train data normalized to 0 mean and unit variance
    height : int

    width : int

    channels : int

    Returns
    -------
    float
        weights normalized to [0, 1]


    """
    rmc_weights = weights.reshape(height, width, channels)
    nmc_weights = (
        rmc_weights - rmc_weights.min()) / (
        rmc_weights.max() - rmc_weights.min())  # Normalize to [0, 1].
    return nmc_weights


def plot_images(
        images,
        r=2,
        c=3,
        title='Multiclass weights',
        figure_name='module_1_assigment_3_Q4_figure.png'):
    """Create image subplots.

    Parameters
    ----------
    images : float
        [n, h, w, c] image tensor
    r : int
        Rows of your image plot
    c : int
        Columns of your image plot
    title : str
        Title of your image plot
    figure_name : str
        Output name of your figure

    Returns
    -------
    (none)

    """
    f = plt.figure()
    plt.suptitle(title)
    for idx, im in enumerate(images):
        plt.subplot(r, c, idx + 1)
        plt.imshow(im)
    plt.savefig(figure_name)
    plt.show()
    plt.close(f)


# Normalize and view your multiclass weights
nweights = [normalize_weights(
    weights=mc_weights[:, idx],
    height=height,
    width=width,
    channels=channels) for idx in range(mc_weights.shape[-1])]
plot_images(nweights)
