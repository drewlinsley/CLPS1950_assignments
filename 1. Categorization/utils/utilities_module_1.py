"""Module 1 utilities."""
import os
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from scipy import misc
from matplotlib import pyplot as plt


def build_m1_model(height, width, channels, path):
    """Build a logistic regression and load saved weights."""
    model = Sequential()
    model.add(
        Dense(
            1,
            activation='sigmoid',
            input_dim=height * width * channels))
    sgd = optimizers.SGD(
        lr=1e-2,
        decay=1e-6,
        momentum=0.9,
        nesterov=True)
    model.compile(
        optimizer=sgd,
        loss='binary_crossentropy')
    model.load(path)
    return model


def derive_labels(files):
    """Find category of every image."""
    cats, dogs = cat_dog_pointers()
    labels = []
    for f in files:
        fname = f.split(os.path.sep)[-1].split('.')[0]
        split_f = fname.split('n')
        if len(split_f) > 1:
            fcat, fit = fname.split('n')
            if fcat in cats.keys() and int(fit) in range(
                    cats[fcat][0], cats[fcat][1]):
                labels += [1]
            elif fcat in dogs.keys() and int(fit) in range(
                    dogs[fcat][0], dogs[fcat][1]):
                labels += [2]
            else:
                print(fcat, fit)
        else:
            labels += [0]
    return np.asarray(labels)


def derive_mc_labels(files, images):
    """Get multiclass labels."""
    files = np.asarray(files)
    images = np.asarray(images)

    # Train multiclass model
    pre_p = np.asarray(
        [int(
            f.split(
                os.path.sep)[-1].split('f')[1].split('t')[0]) for f in files])
    post_p = np.asarray(
        [int(
            f.split(
                os.path.sep)[-1].split('t')[-1].split('n')[0]) for f in files])
    morph = np.asarray(
        [int(
            f.split(
                os.path.sep)[-1].split('n')[-1].split('.')[0]) for f in files])
    pre_p_mask = morph > 7
    post_p_mask = morph <= 7
    pre_p[pre_p_mask] = -1
    post_p[post_p_mask] = -1
    combined_labels = np.copy(pre_p)
    combined_labels[combined_labels == -1] = post_p[pre_p == -1]
    training_split = 0.9  # Hold out this proportion for training
    num_labels = len(combined_labels)
    train_num = np.round(num_labels * training_split).astype(int)
    shuffle_idx = np.random.permutation(num_labels)
    train_idx = shuffle_idx[:train_num]
    test_idx = shuffle_idx[train_num:]
    train_labels = combined_labels[train_idx]
    test_labels = combined_labels[test_idx]
    train_images = images[train_idx]
    test_images = images[test_idx]
    return {
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_images': train_images,
        'test_images': test_images
    }


def normalize_weights(weights, height, width, channels):
    """Reshape and normalize weights."""
    rmc_weights = weights.reshape(height, width, channels)
    nmc_weights = (
        rmc_weights - rmc_weights.min()) / (
        rmc_weights.max() - rmc_weights.min())  # Normalize to [0, 1].
    return nmc_weights


def plot_images(images, r=2, c=3, title='Multi-class weights'):
    """Create image subplots."""
    f = plt.figure()
    plt.suptitle(title)
    for idx, im in enumerate(images):
        plt.subplot(r, c, idx + 1)
        plt.imshow(im)
    plt.show()
    plt.close(f)


def cat_dog_pointers():
    """Cat and dog image labels."""
    cat_labels = {
        'Pf1t0': [1, 15],
        'Pf2t0': [1, 15],
        'Pf2t1': [1, 15],
        'Pf3t0': [8, 15],
        'Pf3t1': [8, 15],
        'Pf3t2': [8, 15],
        'Pf4t0': [8, 15],
        'Pf4t1': [8, 15],
        'Pf4t2': [8, 15],
        'Pf5t0': [8, 15],
        'Pf5t1': [8, 15],
        'Pf5t2': [8, 15]
    }
    dog_labels = {
        'Pf3t0': [1, 8],
        'Pf3t1': [1, 8],
        'Pf3t2': [1, 8],
        'Pf4t0': [1, 8],
        'Pf4t1': [1, 15],
        'Pf4t2': [1, 8],
        'Pf4t3': [1, 15],
        'Pf5t0': [1, 8],
        'Pf5t1': [1, 8],
        'Pf5t2': [1, 8],
        'Pf5t3': [1, 15],
        'Pf5t4': [1, 15]
    }
    return cat_labels, dog_labels


def bucket_ims(files):
    """Bin images according to their doggy-cat continuum."""
    good_names = [
        'Pf3t0',
        'Pf3t1',
        'Pf3t2',
        'Pf4t0',
        'Pf4t2',
        'Pf5t0',
        'Pf5t1',
        'Pf5t2'
    ]
    buckets = {}
    for fi in files:
        buckets[fi] = 0
        for name in good_names:
            if re.search(name, fi):
                num = re.sub('\D', '', re.sub('.*Pf\dt\dn', '', fi))
                buckets[fi] = int(num)
    return buckets


def get_dc_ims(files):
    """Gist: Get dog-cat images with regex."""
    category_regex = '(Pf3t0n\d+)'
    dc_continuum = np.asarray(
        [fi for fi in files if re.search(category_regex, fi) is not None])
    dc_idx = np.argsort(
        [int(re.search('n\d+', x).group()[1:]) for x in dc_continuum])
    sorted_dc = dc_continuum[dc_idx]
    sorted_dc_ims = np.asarray(
        [misc.imread(x) for x in sorted_dc])
    return sorted_dc_ims, sorted_dc


def plot_dc_gradient(
        res_dc_ims,
        sorted_dc,
        dc_probabilities,
        figure_name='module_1_assigment_2_Q1_figure.jpeg'):
    """Gist: Execute operations for plotting the dog-cat gradient."""
    f = plt.figure()
    gdog_idx = range(len(res_dc_ims) // 2)
    plt.scatter(
        gdog_idx,
        dc_probabilities[gdog_idx],
        c='Red',
        label='Dogs')
    gcat_idx = range(len(res_dc_ims) // 2, len(res_dc_ims))
    plt.scatter(
        gcat_idx,
        dc_probabilities[gcat_idx],
        c='Blue',
        label='Cats')
    plt.legend()
    plt.title('Classifier confidence in viewing Dog image.')
    dc_file_names = [x.split(os.path.sep)[-1] for x in sorted_dc]
    dc_file_names = [
        'dog_%s' % x.split('n')[-1]
        if idx < 7
        else 'cat_%s' % x.split('n')[-1]
        for idx, x in enumerate(sorted_dc)]
    plt.xticks(
        range(len(dc_probabilities)),
        dc_file_names,
        rotation=300,
        fontsize=6)
    plt.tight_layout()
    plt.ylabel('Confidence in dog')
    plt.xlabel('Images ranked from most dog (1) to most cat (14)')
    plt.savefig(figure_name)
    plt.show()
    plt.close(f)
