"""Data utilities for CLPS 1950."""


def normalize(x, method, eps=1e-12, mu=None, sd=None):
    """Normalize data matrix x with method.

    Parameters
    ----------
    x : float
        Array of data to normalize
    eps : float
        Constant offset to protect from divide by 0
    mu : float
        Normalizing mean. Calculated from x if not supplied.
    sd : float
        Normalizing std. Calculated from x if not supplied.

    Returns
    -------
    float
        A normalize x
    float
        The normalizing mean
    float
        The normalizing std
    """
    if method == 'zscore':
        if mu is None:
            mu = x.mean(0)
        if sd is None:
            sd = x.std(0)
        nx = (x - mu) / (sd + eps)
        return nx, mu, sd
    else:
        raise NotImplementedError
