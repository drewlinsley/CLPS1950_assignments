"""
## Module 2 assignment 3: Overfitting versus generalization


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 1-6. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_2_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this experiment should be called:

{your_name}_module_2_assignment_3.zip

## Description

You should now have a pretty good idea about how to construct multilayer
neural network models using Keras. You should also have observed that
adding layers to your network improved its performance on classifying
images from the CIFAR-10 image dataset. As discussed in lecture, this
improved performance is owed to the fact that multilayer networks (also
known as multilayer perceptrons; MLPs) can approximate essentially any
decision function (e.g. correctly classifying CIFAR images) given
sufficient training data and capacity (i.e. number of hidden units).

The point of computer vision however is not simply to "fit" a given
dataset but also to "generalize" to new image examples that were not in the
image set used to train a model. How can we ensure that our models
generalize? In other words, how can we control model's overfitting?

This assignment will introduce a variety of methods for controlling
overfitting. You will use Keras (https://keras.io/) to train your models.
Keras is a high-level abstraction over the Tensorflow machine learning
framework that can make it much easier to fit certain kinds of models to
datasets.

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
import numpy as np
from utils import utilities_module_2
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

"""
## Load the images.
"""

x_train, y_train, x_test, y_test, _ = utilities_module_2.unpack_cifar()
# To make computation more efficient, we are going to limit the training and
# test sets to their first 250 images across 4 image categories.
x_train, y_train, keep_cats = utilities_module_2.trim_categories(
    x=x_train,
    y=y_train,
    n=250,
    cats=4)
x_test, y_test, _ = utilities_module_2.trim_categories(
    x=x_test,
    y=y_test,
    n=250,
    keep_cats=keep_cats)


"""
###########################
1. EDIT HERE!
###########################
What is overfitting? It happens when your model's ability to make accurate
decisions on images not in the training dataset starts to decrease. In
other words, a certain amount of training yields good generalization, but
too much can allow your model to overlearn biases in the training set:
that dogs only appear in grass fields, or birds are only ever seen in relief
against the sky.

Overfitting can be observed by comparing a model's fit on the set of images
used for training versus a separate test/validation set of images that is
held out of training.

(Note that in this assignment we do not have distinct validation and test
sets. It is best practice to hold out a subset of your training dataset
as a validation set that will aid in finetuning model operations, such
as early stopping criteria. You should typically pass your test dataset
through your model only after settling on an optimal model.)

Your task is to build a deep model and measure its performance on both the
training and test set.

Q1: Plot the model's loss on the training and test image sets across 10
epochs of training. Around which epoch does your model begin to overfit?

Estimated time: 30 minutes.
"""
ey_train =
ey_test =
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
number, height, width, channels =
epochs =
output_dim =
lr = 0.01
batch_size = 32
steps_per_epoch = number / batch_size


def plot_performance(train_loss, test_loss, image_name):
    """Plot losses on a training versus validation set.

    Parameters
    ----------
    train_loss : list
        List of losses on the train dataset
    test_loss : list
        List of losses on the test dataset
    image_name : str
        Name of your output image

    Returns
    -------
    None

    """
    f, ax1 = plt.subplots()
    ax1.plot(train_loss, 'b-', label='Training loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')
    ax1.plot(test_loss, 'r-', label='Validation loss')
    plt.legend()
    f.tight_layout()
    plt.savefig(image_name)
    plt.show()
    plt.close(f)


# Build your model
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

model = model_builder(x_train, layers)
optim = optimizers.SGD(lr=lr)
model.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])

# Train your model
output = model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
train_loss = output.history['loss']
test_loss = output.history['val_loss']
plot_performance(
    train_loss=train_loss,
    test_loss=test_loss,
    image_name='module_2_assigment_3_Q1_figure.png')


"""
###########################
2. EDIT HERE!
###########################
When training neural models it is impossible to know before hand exactly when
overfitting will set in -- the point at which training and test losses diverge.

Wouldn't it be nice to develop a set of rules that will stop training as soon
as evidence for overfitting starts to accrue? These rules are called
"early stopping criteria".

Q2: What are early stopping criteria that would work on the model you trained
above? Would these criteria hold for other models and dataset? If not, what
are some circumstances in which you can imagine them failing?

Estimated time: 5 minutes.
"""

"""
###########################
3. EDIT HERE!
###########################
Keras makes it simple to create early stopping criteria.

The class below is initialized with parameters that amount to one kind of
early stopping criteria. Go to the following link, adjust the parameters
to the criteria you laid out above in Q2, and demonstrate that your early
stopping criteria successfully stop training around when overfitting sets
in.

https://keras.io/callbacks/#earlystopping

Q3: Provide a plot of your model's training and test losses across training.
This plot should demonstrate that training ends when overfitting sets in.

Estimated time: 20 minutes.
"""

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto')

# Train your model here
es_output = model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[early_stop])
es_train_loss = es_output.history['loss']
es_test_loss = es_output.history['val_loss']


# Plot your model losses here
plot_performance(
    train_loss=es_train_loss,
    test_loss=es_test_loss,
    image_name='module_2_assigment_3_Q3_figure.png')

"""
###########################
4. EDIT HERE!
###########################
Early stopping is a very powerful and computationally cheap method of
regularizing your model to improve its generalization. But this is only one
of many methods for regularizing your neural network model.

Another powerful approach is called dropout. Your model consists neurons, or
computational units, and weights that connect these units together. Dropout
involves randomly turning off these computational units. This is done by setting
a unit's activity to 0 if a random sample from a uniform random distribution is
above a threshold (typically chosen as 0.5; i.e. dropout is sampled from a
Bernoulli distribution or a coin flip). Thus, on every batch of training, a
random set of approximately half of the units in your network will have zero
activity. With this restriction in place, it becomes more difficult for
computation units (and the weights that yield their activity) to overfit to
any particular charactaristics of your dataset.

As an aside, when using dropout, the readout layer of your network becomes
adjusted to the overall smaller magnitude of activity that comes from
network units (i.e. fewer units are on). At test time, however, you want
to take advantage of the full power of your network and turn off dropout.
This means that activity of all units with dropout during training must be
scaled at test time. Fortunately, the dropout formulation used in deep learning
frameworks like Tensorflow and Keras take care of this scaling automatically.

Your task is to train a network with dropout and compare its performance to
a network trained without dropout. Do not use early stopping in this
comparison.

Q4: Make a plot of the dropout model's training and test losses across
training. This plot should compare performance of a model with dropout
to one without dropout. You can either produce two figures or overlay
both model's performances in the same plot. Which model performs better?

Estimated time: 45 minutes.
"""


# Build your model
def model_builder_dropout(x_train, layers, default_dropout_rate=0.5):
    """Build a model from layers to fit x_train.

    Parameters
    ----------
    x_train : float
        [n, h, w, c] tensor of images
    layers : list
        List containing specs for multilayer neural networks
    default_dropout_rate : float
        Default rate of dropout if not specified in a layer

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
        operation = layer.get('operation')
        if operation == 'dense':
            model.add(
                Dense(
                    ,
                    activation=))
        elif operation == 'dropout':
            rate = layer.get('rate', default_dropout_rate)
            model.add(Dropout(rate=rate))
        else:
            raise NotImplementedError  # No other operations
    return model


# In Keras, dropout is its own 'layer'. That means we have to change
# how we're building models. Note the added "operation" field.
dropout_layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
        'operation': 'dense',
        'neurons': ,  # Number of layer neurons
        'activation':  # Activity function for this layer
    },
    {  # First hidden layer
        'operation': 'dropout',
        'rate': 0.5,  # Dropout rate
    },
    {  # Output layer
        'operation': 'dense',
        'neurons': ,  # Number of layer neurons
        'activation':  # Activity function for this layer
    },
]
vanilla_layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
        'operation': 'dense',
        'neurons': ,  # Number of layer neurons
        'activation':  # Activity function for this layer
    },
    {  # Output layer
        'operation': 'dense',
        'neurons': ,  # Number of layer neurons
        'activation':  # Activity function for this layer
    },
]

# Train the dropout model
dropout_output = dropout_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
dropout_train_loss = dropout_output.history['loss']
dropout_test_loss = dropout_output.history['val_loss']

# Train the vanilla model
vanilla_output = vanilla_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
vanilla_train_loss = vanilla_output.history['loss']
vanilla_test_loss = vanilla_output.history['val_loss']

# Plot your model losses here
plot_performance(
    train_loss=dropout_train_loss,
    test_loss=dropout_test_loss,
    image_name='module_2_assigment_3_Q4_dropout_figure.png')
plot_performance(
    train_loss=vanilla_train_loss,
    test_loss=vanilla_test_loss,
    image_name='module_2_assigment_3_Q4_vanilla_figure.png')


"""
###########################
5. EDIT HERE!
###########################
Another popular regularization method is called "weight decay". In a Bayesian
framework this is a prior for "shrinkage" on your model training procedure;
in machine learning popular forms of weight decay are "Tikhonov" or "Ridge"
(in the case of regression).

In each case, the goal is the same: place a constraint on the model during
training that will improve its generalization and control overfitting. In
the case of a single layer perceptron, this would be implemented by adding
an additional term to the loss.

In module 1 assignment 3 we introduced a model that predicts the
category of an image x. This model was defined as:

z = xw + b
y_hat = sigmoid(z)

We optimized this model with a cross entropy loss between the
image's category and the model's prediction of its category.

L = CE(y, y_hat),

where L is the cross entropy loss between an image's category and a model's
prediction of its category. Adding weight decay to this loss involves an
evaluation of the weights used to produce y_hat.

L = CE(y, y_hat) + f(w)

In words, we regularize the model by forcing it to jointly minimize
losses associated with (1) categorization (via the cross entropy error) and
(2) f(w). There are many viable choices for the function to apply to w for
weight decay. A commonly used one is the L2 (euclidean) norm of w.

||w||_2 = sqrt(sum(w**2))

Which yields the following loss.

L = CE(y, y_hat) + alpha*||w||_2

This form of weight decay fights against any single weight becoming an outlier
and tightly overfitting to some spurious characteristic of the dataset. Note
that we also introduce a scalar alpha that will rescale the weight decay into a
similar range as the cross entropy loss. Without this scalar, the weight decay
would dominate the loss (think about the magnitudes of L2 versus cross entropy
losses), and hurt the model's ability to learn from the dataset. This means
that the scale of weight decay is a free parameter that must be tuned by hand
to align with the model's task-related loss.

Your job is to add L2 weight decay to a model and measure its impact on
training.

Train three total models: one without weight decay, one with weight decay set
to a certain strength, and one with weight decay set to a different strength.
Do not enable early stopping or dropout on these models.

Q5: Produce plots showing the impact of weight decay on model training and
testing performance. You can include each model's performance in a single
figure or create three separate plots.

Estimated time: 30 minutes.
"""


# Build your model
def model_builder_l2(x_train, layers):
    """Build a model from layers to fit x_train.

    Parameters
    ----------
    x_train : float
        [n, h, w, c] tensor of images0
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
        wd = layer.get('weight_decay', False)
        if wd:
            model.add(
                Dense(
                    ,
                    activation=))
        else:
            model.add(
                Dense(
                    ,
                    activation=,
                    kernel_regularizer=keras.regularizers.l2(wd)))
    return model


# Note the operation field is gone and replaced with 'weight_decay'.
wd1_layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
    },
    {  # Output layer
    },
]
wd2_layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
    },
    {  # Output layer
    },
]
vanilla_layers = [  # Specify your model with a list of dictionaries.
    {  # First hidden layer
        'neurons': 5000,  # Number of layer neurons
        'activation': 'relu'  # Activity function for this layer
    },
    {  # Output layer
    },
]

# Build and compile your three models:
wd1_model =

wd2_model =

wd3_model =


# Train the models
wd1_output = wd1_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
wd1_train_loss = wd1_output.history['loss']
wd1_test_loss = wd1_output.history['val_loss']
wd2_output = wd2_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
wd2_train_loss = wd2_output.history['loss']
wd2_test_loss = wd2_output.history['val_loss']
vanilla_output = vanilla_model.fit_generator(
    datagen.flow(
        x_train,
        ey_train,
        batch_size=batch_size),
    validation_data=datagen.flow(
        x_test,
        ey_test,
        batch_size=batch_size),
    validation_steps=ey_test.shape[0],
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
vanilla_train_loss = vanilla_output.history['loss']
vanilla_test_loss = vanilla_output.history['val_loss']

# Plot your model losses here
plot_performance(
    train_loss=wd1_train_loss,
    test_loss=wd1_test_loss,
    image_name='module_2_assigment_3_Q5_wd1_figure.png')
plot_performance(
    train_loss=wd2_train_loss,
    test_loss=wd2_test_loss,
    image_name='module_2_assigment_3_Q5_wd2_figure.png')
plot_performance(
    train_loss=vanilla_train_loss,
    test_loss=vanilla_test_loss,
    image_name='module_2_assigment_3_Q5_vanilla_figure.png')

"""
###########################
6. EDIT HERE!
###########################
Putting it all together:

You have now made functions that help you efficiently construct your multilayer
neural networks.

You have also learned about three forms of regularization: (1) early stopping,
(2) dropout, and (3) L2 weight decay.

Now is your chance to put these components together and train the best model
on recognizing CIFAR object images. You will compete by going to the
#module2-cifar-contest channel on slack and posting a figure of your model's
training/testing performance across epochs. When posting your figure, also
mention the max accuracy your model achieved on the test set.

The student with the best performing model will receive 3 points of extra
credit.

Q6: Are the three forms of regularization that you've learned complementary? Or
do certain ones work better than others on CIFAR?
"""
