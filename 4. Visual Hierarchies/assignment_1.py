"""
## Module 4 assignment 1: Creating a visual hierarchy


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 1-5. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_4_assignment_1.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_4_assignment_1.zip

## Description

In this assignment you will significantly increase the expressivity of
multilayer neural networks beyond previous assignments. You will begin
by adding operations to your model that give it "complex"-cell like
responses. In the spirit of the seminal findings of Hubel & Wiesel,
these complex cells will pool over the responses of so-called "simple"
cells in your network. You will compare the performance of models with
complex cells versus those that only have simple cells.

You will also explore the limits of how far a relatively shallow
multilayer perceptron model can go for object recognition. A solution
to this is offered in the form of adding significantly more depth to
your network. This assignment will ultimately have you reappropriate
a state-of-the-art Deep Convolutional Neural Network for object
recognition. Importantly, this powerful model is simply an extension
of the ideas that you have learned throughout this course.

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
from utils import utilities_module_4, py_utils
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

"""
# Load images CIFAR images.
"""
x_train, y_train, x_test, y_test, _ = utilities_module_4.unpack_cifar()
# To make computation more efficient, we are going to limit the training and
# test sets to their first 250 images across 4 image categories.
x_train, y_train, keep_cats = utilities_module_4.trim_categories(
    x=x_train,
    y=y_train,
    n=250,
    cats=4)
x_test, y_test, _ = utilities_module_4.trim_categories(
    x=x_test,
    y=y_test,
    n=250,
    keep_cats=keep_cats)
ey_train = np.squeeze(np.eye(y_train.max() + 1)[y_train])  # GRADE
ey_test = np.squeeze(np.eye(y_test.max() + 1)[y_test])  # GRADE

"""
###########################
1. EDIT HERE!
###########################

Create a convolutional model for CIFAR with a single
hidden layer (input -> hidden -> output).

Q1. Plot your model's performance on the training and validation
sets across training epochs.

Estimated time: 20 minutes.
"""
number, height, width, channels = x_train.shape
epochs = 1  # GRADE
output_dim =
num_filts = 32
filt_height = 13
filt_width = 13
lr = 3e-4
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)  # Note that we use ADAM, which is more
# flexible than SGD.
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False)
datagen.fit(x_train)
conv_model = Sequential()
conv_model.add(
    Conv2D(
        ,
        (, ),
        activation='relu',
        input_shape=(, , )))
conv_model.add(Flatten())
conv_model.add(
    Dense(
        output_dim,
        activation='softmax'))

"""
# Train the model and plot the loss.
"""

"""
###########################
2. EDIT HERE!
###########################

Add a normalization layer to your baseline model architecture.
This will go on top of the convolution layer:
(input -> hidden -> normalization -> output)

Q2. Plot your model's performance on the training and validation
sets across training epochs for the normalization model. How does
normalization affect performance?

Estimated time: 20 minutes.
"""
norm_model = Sequential()
norm_model.add(
    Conv2D(
        ,
        (, ),
        activation='relu',
        input_shape=(, , )))
norm_model.add(utilities_module_4.LRN2D())
norm_model.add(Flatten())
norm_model.add(
    Dense(
        output_dim,
        activation='softmax'))

"""
###########################
3. EDIT HERE!
###########################

Add complex cells to your baseline model and separately, the normalization
model.

For the baseline model this will go on top of the convolution layer:
(input -> hidden -> pooling -> output)

For the normalization model this will go on top of the normalization layer:
(input -> hidden -> normalization -> pooling -> output)


Q3. Plot the training and testing performance of both kinds of complex cell
models. Which performs better on CIFAR?

Estimated time: 20 minutes.
"""
complex_model = Sequential()
complex_model.add(
    Conv2D(
        ,
        (, ),
        activation='relu',
        input_shape=(, , )))
complex_model.add(utilities_module_4.LRN2D())
complex_model.add(MaxPooling2D(pool_size=(2, 2)))
complex_model.add(Flatten())
complex_model.add(
    Dense(
        output_dim,
        activation='softmax'))

"""
###########################
4. EDIT HERE!
###########################

Take your best performing model from above and fit it to the cats and dogs
dataset, introduced below.

Q4. Plot the training and testing performance of the model. Is performance
above chance? If not, why? How can you improve performance?

Estimated time: 20 minutes.
"""

"""
# Load Cats vs. Dogs image dataset. As usual, look at your images before
using them in experiments.
"""
x_train, y_train, x_test, y_test = utilities_module_4.load_cats_dogs(
    os.path.join('data', 'cats_and_dogs', 'PetImages'))
ey_train =
ey_test =
py_utils.plot_im_grd(x_train[:40], title='Dog and cat images')

"""
# Prepare your model for the cats/dogs dataset.
"""
number, height, width, channels = x_train.shape
epochs = 1
output_dim =
num_filts = 32
filt_height = 13
filt_width = 13
lr = 3e-4
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)  # Note that we use ADAM, which is more
# flexible than SGD.
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False)
datagen.fit(x_train)


"""
###########################
5. EDIT HERE!
###########################

Can't solve cats vs. dogs? Go deeper! Build a state-of-the-art deep
convolutional neural network that has convolutional filters tuned
on a large-scale natural image dataset called Imagenet.

You will use an approach called "transfer learning", in which you
add a perceptron on top of pretrained convolutional filters, and
"fine-tune" this architecture to solve a task. This is of course
fine-tuning because the feature detectors of the network are
already complete, but there activities must be remapped to a new
task.

Fine-tune the vgg16 for classifying cats vs. dogs.

Q5. Plot the training and testing performance of the model. How does
this model compare to the models above?

Compare the VGG16 architecture at the link below to the complex
cell network you trained above:

https://arxiv.org/abs/1409.1556

How do these differences in architectures explain the different
performances of each network?

Estimated time: 20 minutes.
"""
preproc_x_train = preprocess_input(x_train.astype(np.float32))
preproc_x_test = preprocess_input(x_test.astype(np.float32))
vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(height, width, channels))
for layer in vgg.layers:
    layer.trainable = False
vgg_classifier = Sequential()
vgg_classifier.add(Flatten(input_shape=[49, 512]))
vgg_classifier.add(Dense(output_dim, activation='softmax'))
vgg = Model(
    inputs=vgg.input,
    outputs=vgg_classifier(vgg.output))
vgg.compile(
    optimizer=optim,
    loss='categorical_crossentropy',
    metrics=['acc'])
cat_dog_vgg = vgg.fit(
    x=,
    y=,
    batch_size=batch_size,
    epochs=epochs)
f, ax = utilities_module_4.plot_performance(
    ,
    ,
    colors=['b', 'r'],
    label='Deep convolutional model')
