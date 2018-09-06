"""
## Module 4 assignment 3: Gradient Images and Adversarial Examples


## Fill in your name here: ______


## Grading

You will be graded on two items that you must turn into canvas:
    (1) A completed version of this assignment. [3 points]
    (2) A PDF with answers to questions in parts 0 and 2. [2 points]
Your completed assignment and PDF response to questions must be combined
into a zip and submitted to canvas for credit. See
[example_submission]Drew_Linsley_module_4_assignment_3.zip for an
example of how you should zip up your assignment and PDF writeup, as well
as an example formatting of the PDF writeup.

Your submission for this assignment should be called:

{your_name}_module_4_assignment_3.zip

## Description

In the last two assignments, you saw how a neural network with convolution
and pooling operations can be a powerful classifier and you measured this power
across layers. But are these networks infallible? In this assignment, you
will try to answer this question by finding "adversarial examples", images
designed to trick neural networks.

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

"""

from keras import backend as K
import numpy as np
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from utils import utilities_module_4
from keras import optimizers
from keras.layers import Flatten, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras import metrics

"""
##########################
1.    EDIT HERE!
##########################

## Load training images, build a two-layer CNN and train it.
"""
x_train, y_train, x_test, y_test, labels = utilities_module_4.unpack_cifar()
labels = np.asarray(labels)

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
number, height, width, channels = x_train.shape
mu = x_train.mean(0, keepdims=True)
sd = x_train.std(0, keepdims=True)
x_train = (x_train - mu) / sd
x_test = (x_test - mu) / sd

# One-hot encode labels and create a data generator
ey_train = 
ey_test =
datagen =   
datagen.fit(x_train)

# Derive the responses
epochs = 5
output_dim =
lr = 3e-4
batch_size = 32
steps_per_epoch = number / batch_size
optim = optimizers.adam(lr=lr)
num_filts = 32
filt_height = 7
filt_width = 7

# Create your conv model
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

# Train the model
conv_output = conv_model.fit_generator()
conv_train_loss = conv_output.history['loss']
conv_test_loss = conv_output.history['val_loss']

"""
Now, we will attempt to explain the classification decisions of our
trained model by determining what parts  of an input image especially
affect the model's output.

Recall that the gradient of a function is an array whose elements
indicate how each function output changes with respect to each function
input. So, if we want to know how each pixel in an image contributes to
our network's decision, we need only calculate the derivative of the loss
function with respect to every input dimension. After all, this will tell us
which pixels increased or decreased the loss and to what degree.

The gradient will tell us how each pixel affects the ouput, so it will itself
be shaped like an image. We call the gradient with respect to the image the
"gradient image".
"""


def gradient_image(dy, dx, image, alpha=1e-5):
    """Calculate a gradient image for a model.

    Gradient of a layer w.r.t. input.


    """
    gradient_image = K.gradients(dy, dx)
    gradient_image /= (
        K.sqrt(
            K.mean(
                K.square(gradient_image))) + alpha)  # L2 normalize

    # Create a wrapper around sess.run that applies the gradientimage op to
    # [model.input]
    input_data = [image]
    iterate = K.function([dx], [gradient_image])
    grads = iterate(input_data)[0].squeeze()  # Apply the function to an image
    return grads


def adversarial_image(image, grads, mu, sd, eps=1e-4, low=0, high=255):
    """Create an adversarial image using the gradient trick.

    Add the gradient of activity w.r.t. the loss
    """
    modulation = np.sign(grads) * eps
    adversarial_image = image + modulation
    unnorm_adv = (adversarial_image * sd) + mu
    clip_adv = np.clip(unnorm_adv, low, high).astype(np.uint8)
    return clip_adv


"""
###############################
2.         EDIT HERE!
###############################

Calculate several gradient images using the gradient_image function.

Q1. Plot and describe each gradient image. Do you see any parts of the images
that tend to contribute to the classification decision either positively
or negatively? Do you think that the visual cortex is sensitive to the same
parts? Why or why not?
"""
selected_images = x_train[:4]
target_layer = conv_model.layers[0].output  # Convolutional activitives
f = plt.figure()
for idx, im in enumerate(selected_images):
    grad_im = gradient_image(
        dy=,
        dx=,
        image=np.expand_dims(im, axis=0))
    grad_im = utilities_module_4.unravel_ravel_normalize_image(grad_im)
    plt.subplot(2, 4, idx + 1)
    plt.title('Image')
    unnorm_im = (im * sd) + mu
    plt.imshow(unnorm_im.squeeze().astype(np.uint8))
    plt.subplot(2, 4, idx + 5)
    plt.title('Gradient image')
    plt.imshow(np.sum(np.abs(grad_im), axis=-1))
plt.show()
plt.close(f)

"""
###########################
3. EDIT HERE!
###########################

Recall that gradient descent iteratively adds/subtracts a gradient
to the parameters of a function until a local extremum is reached.
If the parameters are synaptic weights and the function is a loss, then
gradient descent can be used to train a neural network. In the above question,
the parameters were actually pixels. What would happen if we performed
gradient descent *directly on the image*?

If you iteratively add gradient images to a correctly-classified input image so
that loss *increases*, then eventually the class decision will flip. The new
image, which has a different class decision than the original one, is called an
"adversarial example."

Q2. Complete the code so that it computes adversarial examples with gradient
descent for various epsilon values. Plot the images. For which epsilons are
the resulting images actually adversarial to the originals (i.e., they have
different class decisions according to the model)? For which epsilons would
you say the resulting images are visibly different from the originals? Do
you think adversarial examples exist for biological visual systems? Provide
evidence for/against your position.

"""
image_idx = 0
selected_image = np.expand_dims(x_train[image_idx], axis=0)
target = ey_train[image_idx]
unnorm_im = (x_train[image_idx] * sd) + mu
target_variable = K.variable(target)
loss = metrics.categorical_crossentropy(conv_model.output, target_variable)
target_layer = conv_model.total_loss  # Loss-layer activity
im_yhat = np.argmax(
    conv_model.predict_generator(
        datagen.flow(),
        steps=1))
epsilons = np.linspace(1e-5, 1, 4)
f = plt.figure()
for idx, eps in enumerate(epsilons):
    grad_im = gradient_image(
        dy=loss,
        dx=conv_model.input,
        image=selected_image)
    adv_im = adversarial_image(
        image=selected_image,
        grads=grad_im,
        eps=eps,
        mu=mu,
        sd=sd)
    adv_yhat = np.argmax(
        conv_model.predict_generator(
            datagen.flow(),
            steps=1))
    plt.subplot(2, 4, idx + 1)
    plt.title('Image\nPrediction = %s' % labels[im_yhat])
    plt.imshow(unnorm_im.squeeze().astype(np.uint8))
    plt.subplot(2, 4, idx + 5)
    plt.title('Adversarial image\neps = %s\nPrediction = %s' % (
        eps, labels[adv_yhat]))
    plt.imshow(adv_im.squeeze())
plt.show()
plt.close(f)
