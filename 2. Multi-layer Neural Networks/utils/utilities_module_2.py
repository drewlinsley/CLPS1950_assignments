"""Module 2 utilities."""
import os
import cPickle
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import keras


def visualize_image(image, label):
    """View CIFAR image + label."""
    plt.imshow(image)
    plt.title(label)
    plt.show()


def reshape_image(image):
    """Reshape vectorized CIFAR images to h/w/c."""
    red = image[:1024].reshape(32, 32)
    green = image[1024:2048].reshape(32, 32)
    blue = image[2048:].reshape(32, 32)
    return np.stack((red, green, blue), axis=-1)


def unpickle(filename):
    """Unpickle archived file."""
    with open(filename, 'rb') as fo:
        d = cPickle.load(fo)
    return d


def trim_categories(x, y, n, cats=4, keep_cats=None):
    """Trim a dataset and keep N unique categorical exemplars."""
    if keep_cats is None:
        unique_cats = np.unique(y)
        keep_cats = unique_cats[:cats]
    cat_mask = np.in1d(y, keep_cats)
    assert n < cat_mask.sum(), 'n must be less than %s' % cat_mask.sum()
    masked_y = y[cat_mask]
    masked_x = x[cat_mask]
    return masked_x[:n], masked_y[:n], keep_cats


def unpack_cifar():
    """Prepare CIFAR-10 images for training."""
    im_dir = os.path.join('data')
    train_batches = glob(os.path.join(im_dir, '*data_batch*'))
    test_batch = os.path.join(im_dir, 'test_batch')
    test_dict = unpickle(test_batch)
    test_x = test_dict['data']
    test_y = np.asarray(test_dict['labels']).reshape((-1, 1))
    unpickled = [unpickle(batch) for batch in train_batches]
    train_x = np.concatenate([d['data'] for d in unpickled], axis=0)
    train_y = np.concatenate(
        np.asarray([d['labels'] for d in unpickled]), axis=0).reshape((-1, 1))
    meta = os.path.join(im_dir, 'batches.meta')
    meta = unpickle(meta)
    label_names = meta['label_names']

    # Reshape images into n/h/w/c
    train_x = np.asarray([reshape_image(im) for im in train_x])
    test_x = np.asarray([reshape_image(im) for im in test_x])
    return train_x, train_y, test_x, test_y, label_names
