"""Python utilities for CLPS 1950."""
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt


def status(msg):
    """Declare a status message.

    Parameters
    ----------
    msg : str
    """
    print '*' * 60
    print msg
    print '*' * 60


def plot_im_grd(
        maps,
        title='Cat-dog images',
        rc=None,
        cc=None):
    """Plot images in a grid.

    Parameters
    ----------
    maps : float numpy array
    title : str
    rc : int
    cc : int
    """
    if rc is None:
        rc = np.ceil(np.sqrt(len(maps))).astype(int)
        cc = np.ceil(np.sqrt(len(maps))).astype(int)
    plt.figure(figsize=(8, 8))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rc, cc)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    for idx, im in enumerate(maps):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(
            im.squeeze(),
            cmap='Greys')
    plt.show()
