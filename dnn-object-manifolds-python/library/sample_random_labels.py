"""
Generate random binary labels (+1/-1) for manifold dichotomy tests.

Translated from sample_random_labels.m
"""

import numpy as np


def sample_random_labels(n_objects, random_labeling_type, rng=None):
    """
    Sample binary labels {+1, -1} for n_objects.

    Parameters
    ----------
    n_objects : int
        Number of objects to label.
    random_labeling_type : int
        0 = IID random (each label independently +1 or -1 with equal prob).
        1 = Balanced (exactly half +1, half -1).
        2 = Sparse (one -1, rest +1).
    rng : np.random.Generator or None
        Random number generator.  If None, uses default.

    Returns
    -------
    y : np.ndarray, shape (n_objects,)
        Labels in {+1, -1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    if random_labeling_type == 2:  # Sparse
        y = np.ones(n_objects, dtype=np.float64)
        # MATLAB: randi(N_OBJECTS, 1) — one random integer in 1..N_OBJECTS
        y[rng.integers(n_objects)] = -1
    elif random_labeling_type == 1:  # Balanced
        y = np.ones(n_objects, dtype=np.float64)
        num_neg = int(np.round(n_objects / 2))
        # MATLAB: randperm(N_OBJECTS, round(N_OBJECTS/2))
        neg_indices = rng.choice(n_objects, size=num_neg, replace=False)
        y[neg_indices] = -1
        assert abs(y.sum()) <= 1, "Non balanced labeling"
    else:  # IID
        # MATLAB: 2*randi([0, 1], [1, N_OBJECTS]) - 1
        y = 2.0 * rng.integers(0, 2, size=n_objects) - 1.0

    return y
