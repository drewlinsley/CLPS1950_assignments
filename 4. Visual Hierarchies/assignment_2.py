"""
## Module 4 assignment 2: Measuring accuracy across a visual hierarchy


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in part 1. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_4_assignment_2.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_4_assignment_2.zip

## Description

Deep learning is all the rage, but what does "deep" have to do with it?
In this assignment, you will measure how accuracy in a network changes
as more layers are added to it.

Fill in missing code to complete the assignment. These portions of the code
are denoted with numbered blocks that say "EDIT HERE!".

Estimated time: 30 minutes.

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
from utils import utilities_module_4
from keras.models import Model, Sequential
from keras import applications
from keras.layers import Flatten, Dense
from keras import optimizers

"""
# Load Cats vs. Dogs image dataset. As usual, look at your images before
using them in experiments.
"""
x_train, y_train, x_test, y_test = utilities_module_4.load_cats_dogs(
    os.path.join('data', 'cats_and_dogs', 'PetImages'))
ey_train = np.squeeze(np.eye(y_train.max() + 1)[y_train])  # GRADE
ey_test = np.squeeze(np.eye(y_test.max() + 1)[y_test])  # GRADE
nb_train_samples, height, width, channels = x_train.shape  # GRADE
nb_validation_samples = len(ey_test)
epochs = 1
batch_size =

"""
###########################
1. EDIT HERE!
###########################

Load a VGG16 model for image classification, as you did in Assignment 1.
Measure the effectiveness of its layers (every 4th layer) for object
classification.

Q1. Plot each layer's performance on the validation dog and cat image set.
What trend do you notice across layers? What do you think this says about
how the representation of each image changes across layers? How does this
relate to visual cortex? If possible, refer to some of the literature you
learned about in lecture.

Estimated time: 60 minutes.
"""

"""
# Create a VGG16 model trained on Image
"""
model = applications.VGG16(weights='imagenet')
num_layers = len(model.layers)
layer_names = []
for ll in range(num_layers):
    layer_names.append(model.layers[ll].name)


def layer_classifer(intermediate_model):
    """Create a classifier layer in keras.

    Parameters
    ----------
    intermediate_model : tensor
        Keras model.

    Returns
    -------
    keras tensor
        The compiled model.
    """
    top_model = Sequential()
    top_model.add(Flatten(input_shape=intermediate_model.output_shape[1:]))
    top_model.add(Dense(2, activation='softmax'))
    full_model = Sequential()
    full_model.add(intermediate_model)
    full_model.add(top_model)

    # Only train the final layer
    for layer in full_model.layers[:-1]:
        layer.trainable = False

    # Compile the model
    full_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'])
    return full_model


accuracy_per_depth = []
for depth in range(1, num_layers, 4):

    # Train a classifier for each layer of the model and store
    # performance on the validation image set.
    # You should use the layer_classifer function here

    # Establish the target convolutional layer for training a classifier
    final_layer = model.layers[depth]
    final_layer_name = final_layer.name
    final_layer_output_shape = final_layer.output.shape.as_list()

    # Create a new "intermediate" model
    intermediate_model = Model(
        inputs=model.input,
        outputs=model.get_layer(final_layer_name).output)
    lc = layer_classifer()  # Insert your intermediate model

    # fine-tune the model
    layer_perf = lc.fit(
        x=,
        y=,
        batch_size=batch_size,
        epochs=epochs)
    max_accuracy_for_this_layer = 
    accuracy_per_depth += [max_accuracy_for_this_layer]
