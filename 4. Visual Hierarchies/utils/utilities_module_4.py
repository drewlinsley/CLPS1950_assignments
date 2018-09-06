"""Module 2 utilities."""
from __future__ import division
import os
import cPickle
import numpy as np
from scipy import misc
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import gridspec
from numpy.fft import fft2, fftshift
from math import pi
from scipy.optimize import fmin_powell
from keras import backend as K
from keras.engine.topology import Layer


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


def normalize_to_zero_one(x, axis=0):
    """Normalize x to [0, 1].

    Parameters
    ----------
    x : float
        Array of floats

    Returns
    -------
    numpy array
        x normalized to [0, 1]

    """
    xmin = x.min(axis, keepdims=True)
    xmax = x.max(axis, keepdims=True)
    return (x - xmin) / (xmax - xmin)


def unravel_ravel_normalize_image(x):
    """Unravel image, normalize it, then reshape it."""
    xs = x.shape
    rx = unravel(x)
    nx = normalize_to_zero_one(rx, axis=1)
    return ravel(nx, xs)


def alexnet_mean():
    """BGR mean for alexnet."""
    return np.asarray([123.68, 116.779, 103.939])


def alexnet_process_image(im):
    """Subtract mean and convert to BGR."""
    # ILSVRC_mean = alexnet_mean()
    # im -= ILSVRC_mean.astype(im.dtype)
    # im = im[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    # return im / 255.
    im = im.astype(np.float32)
    im /= 255.
    im = im[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    im -= im.mean()
    return im


def normalize_ims(ims):
    """Loop through ims and normalize each."""
    nims = []
    for im in ims:
        nims += [alexnet_process_image(im)]
    return np.asarray(nims)


def proc_and_load_images(path):
    """Process and load an image for alexnet."""
    im = misc.imread(path).astype(np.float32)
    im = misc.imresize(im, 0.75)
    return alexnet_process_image(im)


def unravel(x):
    """Unravel tensor to matrix."""
    return x.reshape(x.shape[0], -1)


def ravel(x, shape):
    """Reshape matrix to tensor."""
    return x.reshape(shape)


def load_normalized_filters(path, layer='conv1'):
    """Plot losses on a training versus validation set.

    Parameters
    ----------
    path : str
        Path to a numpy containing a dictionary of filters

    Returns
    -------
    numpy array
        A tensor of normalized filter weights

    """
    assert os.path.exists(path), '%s does not exist.' % path
    filters, biases = np.load(path).item()[layer]
    filters = filters.transpose(3, 0, 1, 2)  # N first
    nf = ravel(
        normalize_to_zero_one(unravel(filters), axis=1),
        shape=filters.shape)
    return nf, filters, biases


def plot_mosaic(
        maps,
        title='Mosaic',
        rows=None,
        columns=None):
    """Plot a mosaic of images."""
    if rows is None:
        rows = np.ceil(np.sqrt(len(maps))).astype(int)
        columns = np.ceil(np.sqrt(len(maps))).astype(int)
    f = plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rows, columns)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    for idx, im in enumerate(maps):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(im.squeeze(), cmap='Greys')
    plt.show()
    plt.close(f)


def plot_performance(
        train_loss=None,
        test_loss=None,
        colors=['b', 'r'],
        label='',
        desc='loss',
        ax=None,
        f=None):
    """Plot losses on a training versus validation set.

    Parameters
    ----------
    train_loss : list
        List of losses on the train dataset
    test_loss : list
        List of losses on the test dataset
    colors : list
        List of colors for plotting
    label : list
        Labels for the plot legends

    Returns
    -------
    None

    """
    if f is None and ax is None:
        f, ax = plt.subplots()
    train_label = '%s training %s' % (label, desc)
    if train_loss is not None:
        ax.plot(train_loss, '%s-' % colors[0], label=train_label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color=colors[0])
    # ax1.tick_params('y', colors=colors[0])
    val_label = '%s validation %s' % (label, desc)
    if test_loss is not None:
        ax.plot(test_loss, colors[1], label=val_label)
    plt.legend()
    f.tight_layout()
    return f, ax


def gabor_kern(params, shape):
    """Produces a gabor kernel.

    *shape*: Tuple: (*height*, *width*)
    *params*: array containing: (*x0*, *y0*, *sigma_x*, *sigma_y*,
    *psi*, *tau*, *k*, *A*). Where *psi* is the wave vector angle
    and *tau* the phase. Consult the group wiki for a detailed
    description.

    Returns an array of shape *shape* containing the generated gabor.

    """

    # Extract parameters
    x0 = params[0:2]
    sigma_unr = np.array([[1 / params[2]**2, 0], [0, 1 / params[3]**2]])
    psi, tau = params[4:6]
    k0 = params[6]
    A = params[7]
    h, w = shape
    R0 = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    sigma_r = np.dot(np.transpose(R0), np.dot(sigma_unr, R0))
    k_r = np.dot(np.transpose(R0), np.array([k0, 0]))
    gx, gy = np.ogrid[0:h, 0:w]
    gx -= x0[0]
    gy -= x0[1]
    e = sigma_r[
        1, 1] * gx * gx + (
            sigma_r[0, 1] + sigma_r[1, 0]) * gx * gy + sigma_r[0, 0] * gy * gy

    angular = (k_r[1] * gx + k_r[0] * gy) + tau
    gabor_kern = (A * np.exp(-0.5 * e)) * (np.cos(angular))
    return gabor_kern


def objective_function(to_estimate, *args):
    """
    """
    gabor_shape, data = args
    return np.sum(np.abs((gabor_kern(to_estimate, gabor_shape) - data)))


def check_bounds(a, b, side):
    """
    """
    if np.sum(np.array([a, b]) <= np.float(side - 1)) < 2:
        return False
    elif np.sum(np.array([a, b]) >= 0.) < 2:
        return False
    else:
        return True


def best_approx(a, b, side):
    """
    """
    if a < 0.:
        ret_a = 0
    elif a > (side - 1):
        ret_a = side - 1
    else:
        ret_a = a
    if b < 0.:
        ret_b = 0
    elif b > (side - 1):
        ret_b = side - 1
    else:
        ret_b = b
    return [ret_a, ret_b]


def gabor_fit(data):
    """
    """
    gabor_shape = data.shape
    h, w = gabor_shape
    if h != w:
        raise ValueError(
            "Sqare input expected (width == height)")
    max_pos = np.argmax(np.abs(data))

    x0 = max_pos // h
    x1 = max_pos % h
    mod_trans = np.abs(fft2(data))

    mod_trans_s = fftshift(mod_trans)
    max_vec = np.argsort(mod_trans_s.flatten())[-2:]

    delta_ver = -0.5 * (max_vec[0] // w - max_vec[1] // w)
    delta_hor = 0.5 * (max_vec[0] % w - max_vec[1] % w)
    k_x = (2 * pi * delta_ver) / w
    k_y = (2 * pi * delta_hor) / w
    if k_y < 0:
        k_x = -k_x
        k_y = -k_y

    k0 = np.sqrt(k_x**2 + k_y**2)

    if k_y != 0.:
        psi = np.arctan(k_x / k_y)
    else:
        psi = pi / 2.

    V_1 = data[x0, x1]
    add_x0 = np.int(np.round((pi / 2.) * (k_x / (k0**2))))
    add_x1 = np.int(np.round(-(pi / 2.) * (k_y / (k0**2))))
    if not check_bounds(x0 + add_x0, x1 + add_x1, h):
        add_x0 = -add_x0
        add_x1 = -add_x1
        if not check_bounds(x0 + add_x0, x1 + add_x1, h):
            a, b = best_approx(x0 + add_x0, x1 + add_x1, h)
            V_2 = data[a, b]
        else:
            V_2 = -data[x0 + add_x0, x1 + add_x1]
    else:
        V_2 = data[x0 + add_x0, x1 + add_x1]

    add2_x0 = -np.int(np.round(2 * pi * (k_x / (k0**2))))
    add2_x1 = np.int(np.round(2 * pi * (k_y / (k0**2))))

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = -add2_x0
        add2_x1 = -add2_x1
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            a, b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
            V_3 = data[a, b]
        else:
            V_3 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_3 = data[x0 + add2_x0, x1 + add2_x1]

    if np.sign(V_3) != np.sign(V_1):
        V_3 = -V_3
    factor = pow((V_3 / V_1), 1. / 16.)
    V_2 = V_2 / factor
    A = np.sqrt(V_1**2 + V_2**2)

    # An the angle by: (although it's not giving an accurate estimate)
    # There might be a bug in this estimate.
    if V_1 != 0:
        tau = np.mod(np.arctan(V_2 / V_1) - (k_x * x0 + k_y * x1), 2 * pi)
        if tau > pi:
            tau -= pi
    else:
        tau = np.mod(pi / 2. - (k_x * x0 + k_y * x1), 2 * pi)
        if tau > pi:
            tau -= pi

    V_1 = data[x0, x1]
    add2_x0 = -np.int(np.round(4 * pi * (k_x / (k0**2))))
    add2_x1 = np.int(np.round(4 * pi * (k_y / (k0**2))))

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = np.int(np.round(0.5 * add2_x0))
        add2_x1 = np.int(np.round(0.5 * add2_x1))
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            add2_x0 = -add2_x0
            add2_x1 = -add2_x1
            if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
                a, b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
                V_2 = data[a, b]
            else:
                V_2 = data[x0 + add2_x0, x1 + add2_x1]
        else:
            V_2 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_2 = data[x0 + add2_x0, x1 + add2_x1]

    if np.sign(V_2) != np.sign(V_1):
        V_2 = -V_2

    add3_x0 = np.int(np.round(6 * (k_y)))
    add3_x1 = np.int(np.round(6 * k_x))
    if not check_bounds(x0 + add3_x0, x1 + add3_x1, h):
        a, b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
        V_3 = data[a, b]
    else:
        V_3 = data[x0 + add3_x0, x1 + add3_x1]

    if np.sign(V_3) != np.sign(V_1):
        V_3 = -V_3

    sigma_x = (4 * pi / k0) * np.sqrt(1. / (-2. * np.log(V_2 / V_1)))
    sigma_y = np.sqrt(-18. / np.log(V_3 / V_1))

    params = np.empty(8)
    params[0] = x0
    params[1] = x1
    params[2] = sigma_x
    params[3] = sigma_y
    params[4] = psi
    params[5] = tau
    params[6] = k0
    params[7] = A
    num_iter_2 = 5

    min_params = np.empty(8)
    min_val = np.inf
    for j in xrange(-3, 4):
        for k in xrange(-3, 4):
            for i in xrange(num_iter_2):
                params_ = params.copy()
                params_[0] = params[0] + sigma_x / 2 * j
                params_[1] = params[1] + sigma_y / 2 * k
                params_[5] = 2 * pi / num_iter_2 * i

                try:
                    cur_params = fmin_powell(
                        objective_function,
                        params_,
                        (gabor_shape, data),
                        disp=0)
                    cur_val = objective_function(cur_params, gabor_shape, data)

                    if cur_val < min_val:
                        min_params = cur_params
                        min_val = cur_val
                except Exception as e:
                    print e

    min_params[5] = np.mod(min_params[5], 2 * pi)
    return min_params


def fit_conv(params, error_vector, image_size):
    """
    Filter results so that only "sensible" gabor-prarams survive.
    """
    para = params.copy()
    error_vec = error_vector.copy()
    bound = (image_size)**2
    for i in (2, 3, 6):
        para[:, i] = np.abs(para[:, i])
        slicer = para[:, i] < bound
        para = para[slicer, :]
        error_vec = error_vector[slicer]

    para[:, 4] = np.mod((para[:, 4] + pi / 2.), pi)
    para[:, 5] = np.mod(para[:, 5], 2 * pi)
    para[:, 6] = (para[:, 6]) / (2 * pi)
    return (para, error_vec)


def load_cats_dogs(path, wc='*.jpg', limit=50, train=0.9, crop=(224, 224)):
    """Load cats and dogs dataset."""
    cats = glob(os.path.join(path, 'Cat', wc))[:1000]
    dogs = glob(os.path.join(path, 'Dog', wc))[:1000]

    def load_im(x, crop):
        """Load an image."""
        try:
            im = misc.imread(x)
            y, x, _ = im.shape
            startx = x // 2 - (crop[1] // 2)
            starty = y // 2 - (crop[0] // 2)
            crop_im = im[starty:starty + crop[1], startx:startx + crop[0]]
            if crop_im is not None:
                return crop_im
        except:
            print 'Found bad image. Skipping.'

    def check_ims(ims, crop):
        """Check that ims satisfied the cropping condition."""
        out_ims = []
        for im in ims:
            try:
                if im is not None and im.shape[0] == crop[0] and im.shape[1] == crop[1]:
                    out_ims += [im]
            except:
                print 'Found bad image. Skipping.'
        return out_ims

    cat_ims = [load_im(x=x, crop=crop) for x in cats]
    dog_ims = [load_im(x=x, crop=crop) for x in dogs]
    cat_ims = check_ims(cat_ims, crop)
    dog_ims = check_ims(dog_ims, crop)
    cat_ims = [
        x for x in cat_ims if x.shape[0] == crop[0] and x.shape[1] == crop[1]]
    dog_ims = [
        x for x in dog_ims if x.shape[0] == crop[0] and x.shape[1] == crop[1]]
    cv_split = np.round(len(cat_ims) * train).astype(int)
    cat_train = cat_ims[:cv_split]
    cat_test = cat_ims[cv_split:]
    dog_train = dog_ims[:cv_split]
    dog_test = dog_ims[cv_split:]
    x_train = np.concatenate((cat_train, dog_train))
    x_test = np.concatenate((cat_test, dog_test))
    y_train = np.hstack((np.zeros(len(cat_train)), np.ones(len(dog_train))))
    y_test = np.hstack((np.zeros(len(cat_test)), np.ones(len(dog_test))))
    train_shuffle = np.random.permutation(len(x_train))
    test_shuffle = np.random.permutation(len(x_test))
    x_train = x_train[train_shuffle]
    y_train = y_train[train_shuffle].reshape(-1, 1).astype(int)
    x_test = x_test[test_shuffle]
    y_test = y_test[test_shuffle].reshape(-1, 1).astype(int)
    assert len(x_train), 'Could not find pet images in %s' % path
    assert len(x_test), 'Could not find pet images in %s' % path
    return x_train, y_train, x_test, y_test


class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate(
            [
                extra_channels[:, :half_n, :, :],
                input_sqr,
                extra_channels[:, half_n + ch:, :, :]],
            axis=1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[
        :, centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]
