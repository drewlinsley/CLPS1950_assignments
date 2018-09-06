"""
## Module 2 assignment 2: Multi-layer neural networks


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 2, 3, and 4. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_2_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_2_assignment_2.zip

## Description

In Module 1 you trained a neural network to classify relatively simple
images of dogs and cats overlaid against empty backgrounds. This simplicity
was validated by visualizing the network model weights: the presence or
absence of pixels at different locations drove the model towards cat or
dog classifications.

Does this approach hold for more complex datasets?

In this assignment you will compare the effectiveness of a single-layer
neural network to a multilayer neural network in classifying images of
objects from the classic CIFAR-10 dataset.

This dataset contains 50,000 images from the TinyImages dataset
(see: http://groups.csail.mit.edu/vision/TinyImages/), each of which belongs
to one of ten object categories. An additional 10,000 images are available
for testing your model.

You will use Keras (https://keras.io/) to train your models. Keras is a
high-level abstraction over the Tensorflow machine learning framework.

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

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
import random
import numpy as np
from utils import utilities_module_2
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

"""
## Load the images.
"""

x_train, y_train, x_test, y_test, labels = utilities_module_2.unpack_cifar()

"""
## Visualize your data.
"""


def view_images(num_images=10):
    """View a few of the CIFAR images and their labels."""
    for k in range(num_images):
        i = random.randrange(len(y_train))
        image = x_train[i]
        label = y_train[i]
        label = labels[label[0]]
        utilities_module_2.visualize_image(image, label)


view_images(5)

"""
###########################
1. EDIT HERE!
###########################
Your first task is to prepare your labels for training in Keras.
CIFAR-10 is an image dataset that samples from 10 unique object
categories.

As discussed in module 1 assignment 3, a common approach to
multicategory classification is to use a loss function like
categorical cross entropy. To prepare your data for this you must
create a one-hot encoding of it.

ey_train should have the shape [50000, 10]
ey_test should have the shape [10000, 10]


Estimated time: 2 minutes.
"""
ey_train =
ey_test =

"""
## Build an image processing pipeline with Keras.

The ImageDataGenerator class gives you a large number of options
for processing your image data before it is passed to your neural
network model. In the previous module's assignments, you applied a
"zscore" to each pixel location across images in the cat and dog
dataset. This is equivalent to setting "featurewise_center" and
"featurewise_std_normalization" to True in the below function.

As an additional exercise, explore the Keras API to understand what the
other ImageDataGenerator methods do: https://keras.io/preprocessing/image/
"""

datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None)
datagen.fit(x_train)  # Fit the datagen to training image statistics.
"""
This may take a while. Datagen only works with tensors (e.g. n/h/w/c).

The point of this batch-data loader is to let you train your model in
"batches" of images rather than the entire dataset at once. In module 1
assignment 3 you implemented a gradient descent procedure where you
updated your weights based on error derived from all of the images in
the training set at once. The efficiency of training improves by training
on batches instead of the entire image set. This approach is called
Stochastic Gradient Descent, and is implemented below.
"""

"""
###########################
2. EDIT HERE!
###########################
Use Keras to train the same model you used in Module 1 assignment 3.

Your task is to specify the parameters necessary for training this model.

Q1: Provide a plot of your model's loss and accuracy during training.
Define its axes and discuss what it means: is your model well-suited
for classifying the CIFAR-10 dataset? What is its average accuracy on the
CIFAR test image set and how does that compare to chance?

Estimated time: 15 minutes.
"""
number, height, width, channels =
epochs = 3
output_dim =
lr = 0.01
batch_size = 32
steps_per_epoch = number / batch_size

# Build your Keras model
model = Sequential()
model.add(
    Reshape(  # Reshape an example from a tensor to a vector
        (height * width * channels,),
        input_shape=(height, width, channels)))
model.add(
    Dense(
        output_dim,
        activation='softmax'))
optim = optimizers.SGD(lr=lr)
model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])


# Save the history of losses in your Keras model
class History(keras.callbacks.Callback):
    """Save losses at each batch with a Keras model callback function."""
    def on_train_begin(self, logs={}):
        """Initialize an empty list for storing model losses."""
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        """Append a loss after each batch ends."""
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


# Train your model
history = History()
model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[history])
losses = history.losses
accuracies = history.accuracies

# Plot the loss and categorization accuracy on the training set across epochs
f, ax1 = plt.subplots()
ax1.plot(losses, 'b-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(accuracies, 'r-', alpha=0.5)
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params('y', colors='r')
f.tight_layout()
plt.title('One-layer network loss and categorization accuracies on CIFAR')
plt.savefig('module_2_assigment_2_Q1_figure.png')
plt.show()
plt.close(f)

# Test your model
test_batch =  # Choose a batch size of your test images that is divisible by
# x_test.shape[0]
test_loss, test_accuracy = model.evaluate_generator(
    generator=datagen.flow(
        x_test,
        ey_test,
        batch_size=test_batch),
    steps=x_test.shape[0] / test_batch)
print('Model test image loss: %s | test image accuracy: %s' % (
    test_loss,
    test_accuracy))


"""
###########################
3. EDIT HERE!
###########################
Use Keras to create a multilayer model.

To do this, you will build a function
that takes as input various parameters that are necessary for specifying your
model. It will output a model with those specs. This approach reduces the
potential for bugs in your code.

Q2: Create a plot of the loss while training this new multilayer model. Overlay
the accuracy for each epoch of training. How does this performance compare to
the single-layer model you created above? How does depth impact performance?

Estimated time: 30 minutes.
"""


def model_builder(x_train, layers):
    """Build a model from layers to fit x_train.

    Parameters
    ----------
    x_train : float
        [n, h, w, c] tensor of images
    layers : list
        List containing specs for multilayer neural networks

    Returns
    -------
    object
        A Keras model
    """
    number, height, width, channels =

    # Build your Keras model
    model = Sequential()
    model.add(
        Reshape(  # Reshape an example from a tensor to a vector
            (,),
            input_shape=()))
    for layer in layers:
        model.add(
            Dense(
                ,
                activation=))
    return model


layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
        'neurons': ,  # Number of layer neurons
        'activation': 'relu'  # Activity function for this layer
    },
    {  # Output layer
        'neurons': ,  # Number of layer neurons
        'activation':  # Output activity function that aligns with line 331.
    },
]

multilayer_model = model_builder(x_train, layers)  # Grade
multilayer_model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])

# Train your model
multilayer_history = History()
multilayer_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[multilayer_history])
multilayer_losses = multilayer_history.losses
multilayer_accuracies = multilayer_history.accuracies

# Plot your training loss and accuracy across epochs
f, ax1 = plt.subplots()
ax1.plot(multilayer_losses, 'b-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(multilayer_accuracies, 'r-')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params('y', colors='r')
f.tight_layout()
plt.title(
    '%s-layer network loss and categorization accuracies on CIFAR-10' % len(
        layers))
plt.savefig('module_2_assigment_1_Q2_figure.png')
plt.show()
plt.close(f)


"""
###########################
4. EDIT HERE!
###########################
Systematically measure the impact of depth on learning the CIFAR-10

Create a new function that takes a list of layers and returns model accuracy on
the test set of CIFAR images.

Q3: Compare at least three different model architectures, and discuss how
increasing the number of layers influences performance. Is there a monotic
increase in performance or is there a tradeoff between number of layers and
performance? What is the "sweet spot"? Include your plot in your response.

Estimated time: 45 minutes.
"""


def evaluate_models(
        x_train,
        ey_train,
        x_test,
        ey_test,
        models,
        model_params):
    """Evaluate performance of models on held-out images.

    Parameters
    ----------
    x_train : float
        [n, h, w, c] tensor of images
    ey_train : int
        [n, c] one-hot encoded matrix of labels
    x_test : float
        [m, h, w, c] tensor of images
    ey_test : int
        [m, c] one-hot encoded matrix of labels
    models : list
        List of lists containing specs for multilayer neural networks
    model_params: dict
        Dictionary of model hyperparameters

    Returns
    -------
    list
        The losses of each model on the test image set
    list
        The accuracies of each model on the test image set
    """
    eval_loss = []
    eval_accuracy = []
    for idx, layers in enumerate(models):
        print('_' * 30)
        print('Working on model %s/%s' % (idx + 1, len(models)))
        print('_' * 30)
        model_history = History()
        model = model_builder(,)
        model.compile(
            optimizer=model_params['optim'],
            loss='categorical_crossentropy',
            metrics=['acc'])
        model.fit_generator(
            datagen.flow(
                ,
                ,
                batch_size=model_params['batch_size']),
            steps_per_epoch=model_params['steps_per_epoch'],
            epochs=model_params['epochs'],
            callbacks=[model_history])
        test_loss, test_accuracy = model.evaluate_generator(
            generator=datagen.flow(
                ,
                ,
                batch_size=model_params['test_batch']),
            steps=x_test.shape[0] / model_params['test_batch'])
        eval_loss += [test_loss]
        eval_accuracy += [test_accuracy]
        print('Model %s: test image loss: %s | test image accuracy: %s' % (
            idx + 1,
            test_loss,
            test_accuracy))
    return eval_loss, eval_accuracy


models = [
    [  # Model 1
        {
        },
    ],
    [  # Model 2
        {
        },
    ],
    [  # Model 3
        {
        },
    ]
]
model_params = {
    'epochs': epochs,
    'batch_size': batch_size,
    'steps_per_epoch': steps_per_epoch,
    'test_batch': test_batch,
    'optim': optim
}
model_losses, model_accuracies = evaluate_models(
    x_train=,
    ey_train=,
    x_test=,
    ey_test=,
    models=,
    model_params=)

# Plot training loss and accuracy of the models
bar_width = 0.3
f, ax1 = plt.subplots()
loss_inds = range(0, len(model_losses) * 3, len(model_losses))
acc_inds = range(1, len(model_losses) * 3, len(model_losses))
ax1.bar(loss_inds, model_losses, bar_width, color='b')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Losses', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.bar(acc_inds, model_accuracies, bar_width, color='r')
ax2.set_ylabel('Accuracies', color='r')
ax2.tick_params('y', colors='r')
f.tight_layout()
plt.title(
    'Comparison of %s models on CIFAR-10' % len(models))
loss_inds = np.arange(len(model_losses)).repeat(2)
model_names = [x if idx % 2 else '' for idx, x in enumerate(loss_inds)]
plt.xticks([0, 3.5, 7], ['Model 1', 'Model 2', 'Model 3'])
plt.savefig('module_2_assigment_2_Q3_figure.png')
plt.show()
plt.close(f)
