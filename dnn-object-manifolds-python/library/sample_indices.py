import numpy as np


def sample_indices(N, K, R=1, rng=None):
    """Sample R random subsets of K indices from range(N) without replacement.

    Parameters
    ----------
    N : int
        Size of the population (indices 0 .. N-1).
    K : int
        Number of indices to draw per sample.
    R : int, optional
        Number of independent samples (default 1).
    rng : numpy.random.Generator or None
        Random number generator. If None, a new default Generator is created.

    Returns
    -------
    samples : ndarray of shape (R, K)
        Each row contains K unique indices drawn from range(N).
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = np.zeros((R, K), dtype=int)
    for r in range(R):
        samples[r, :] = rng.choice(N, size=K, replace=False)
    return samples
