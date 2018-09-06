"""Data utilities for CLPS 1950."""
import numpy as np


def normalize(x, method, eps=1e-12, mu=None, sd=None):
    """Normalize data matrix x with method.

    Parameters
    ----------
    x : float numpy array
    eps : float
    mu : float array
    sd : float array
    """
    if method == 'zscore':
        if mu is None:
            mu = x.mean(0)
        if sd is None:
            sd = x.std(0)
        nx = (x - mu) / (sd + eps)
        return nx, mu, sd
