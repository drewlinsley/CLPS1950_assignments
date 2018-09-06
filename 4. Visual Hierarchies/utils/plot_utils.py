"""Plotting utilities for CLPS1950."""
import os
import numpy as np
from matplotlib import pyplot as plt


def plot_probs(
        probabilities,
        labels,
        accuracy,
        losses,
        files,
        keyword):
    """Plot classifier probabilities versus ground truth labels.

    Parameters
    ----------
    probabilities : float numpy array
    labels : int numpy array
    accuracy : float
    losses : float
    files : list of str
    keyword : str
    """
    f = plt.figure()
    plt.plot(
        range(len(probabilities)),
        labels,
        'r',
        label='%s labels' % keyword)
    plt.plot(
        range(len(probabilities)),
        probabilities,
        'b--',
        label='%s probabilities')
    plt.legend()
    plt.title('%s accuracy: %s, model loss: %s' % (
        keyword,
        accuracy,
        losses.history['loss'][-1]))
    proc_files = [x.split(os.path.sep)[-1] for x in files]
    proc_files = [
        'dog_%s' % x.split('n')[-1]
        if labels[idx]
        else 'cat_%s' % x.split('n')[-1]
        for idx, x in enumerate(proc_files)]
    plt.xticks(
        range(len(probabilities)),
        files,
        rotation=300,
        fontsize=6)
    plt.tight_layout()
    plt.ylabel('Probability of dog image')
    plt.xlabel('Images ranked from most cat (1) to most dog (14)')
    plt.show()
    plt.close(f)


def m1_images(
        ims,
        probabilities,
        labels,
        shape,
        keyword='Train'):
    """Plot the images for which the classifier is most and least confident.

    Parameters
    ----------
    probabilities : float numpy array
    ims : float numpy array
    labels : int numpy array
    shape : int tuple
    keyword : str
    """
    cat_ims = ims[labels == 0]
    dog_ims = ims[labels == 1]
    least_cat = cat_ims[np.argmin(probabilities[labels == 0])]
    most_cat = cat_ims[np.argmax(probabilities[labels == 0])]
    least_dog = dog_ims[np.argmin(probabilities[labels == 1])]
    most_dog = dog_ims[np.argmax(probabilities[labels == 1])]
    f, ax = plt.subplots(1, 4)
    ax[0].imshow(least_cat.reshape(shape), cmap='Greys')
    ax[0].set_title('Cattiest cat')
    ax[1].imshow(most_cat.reshape(shape), cmap='Greys')
    ax[1].set_title('Doggiest cat')
    ax[2].imshow(least_dog.reshape(shape), cmap='Greys')
    ax[2].set_title('Doggiest dog')
    ax[3].imshow(most_dog.reshape(shape), cmap='Greys')
    ax[3].set_title('Cattiest dog')
    plt.suptitle('Classifier confidence in %s images' % keyword)
    plt.show()
    plt.close(f)
