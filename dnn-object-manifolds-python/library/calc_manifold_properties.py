"""
Calculate manifold geometric properties and predicted capacity.

Translated from calc_manifold_properties.m
"""

import numpy as np
import warnings

from library.theory_alpha0 import theory_alpha0_cached


def calc_manifold_properties(cF, center_norm, N_RANDOM_PROJECTIONS=1000, kappa=0, rng=None):
    """
    Calculate manifold properties and predicted capacity using an iterative method.

    Parameters
    ----------
    cF : np.ndarray, shape (N_NEURONS, M)
        Center-subtracted features for one manifold.
    center_norm : float
        Norm of the manifold center.
    N_RANDOM_PROJECTIONS : int, optional
        Number of random projection directions (default 1000).
    kappa : float, optional
        Margin parameter (default 0).
    rng : np.random.Generator or None, optional
        Random number generator for reproducibility.

    Returns
    -------
    mean_half_width1 : float
    mean_argmax_norm1 : float
    mean_half_width2 : float
    mean_argmax_norm2 : float
    effective_dimension : float
    effective_dimension2 : float
    alphac_hat : float
    """
    if rng is None:
        rng = np.random.default_rng()

    TOLERANCE = 0.05 / center_norm
    MAX_ITER = 100

    assert center_norm > 0, 'Norm must be possitive'
    N_NEURONS = cF.shape[0]

    # Scale the data with the size of the centers
    cF = cF / center_norm
    kappa = kappa / center_norm

    w = rng.standard_normal((N_RANDOM_PROJECTIONS, N_NEURONS))
    w_norm = np.sqrt(np.sum(w ** 2, axis=1))

    wc = w @ cF  # (N_RANDOM_PROJECTIONS, M)
    max_random_projections_margin = np.max(wc, axis=1)
    argmax_random_projections_margin = np.argmax(wc, axis=1)

    s0 = cF[:, argmax_random_projections_margin].T  # (N_RANDOM_PROJECTIONS, N_NEURONS)
    s0_norm_square = np.sum(s0 ** 2, axis=1)
    ws0 = np.sum(w * s0, axis=1)

    # assert_warn: warn instead of raising
    err_val = np.max(np.abs(ws0 - max_random_projections_margin))
    if err_val >= 1e-5:
        warnings.warn(f'{err_val:.1e}')

    mean_half_width1 = np.mean(ws0)
    mean_argmax_norm1 = np.mean(s0_norm_square)

    # Iterative refinement
    eta = 0.1  # eta should be small
    error = 1.0
    iter_ = 0
    error_results = np.zeros((MAX_ITER, N_RANDOM_PROJECTIONS))

    while error > TOLERANCE and iter_ < MAX_ITER:
        iter_ += 1  # MATLAB increments at top of loop (1-based)

        z0 = (ws0 + kappa) / (1 + s0_norm_square)
        dw = w - z0[:, np.newaxis] * s0
        dwc = dw @ cF  # (N_RANDOM_PROJECTIONS, M)
        max_random_projections_margin = np.max(dwc, axis=1)
        argmax_random_projections_margin = np.argmax(dwc, axis=1)

        error = np.max(max_random_projections_margin - np.sum(dw * s0, axis=1))
        error_results[iter_ - 1, :] = max_random_projections_margin - np.sum(dw * s0, axis=1)

        if iter_ > 1 and np.mean(error_results[iter_ - 1, :]) > np.mean(error_results[iter_ - 2, :]):
            eta = 0.8 * eta

        if error > TOLERANCE:
            s1 = cF[:, argmax_random_projections_margin].T
            s0 = (1 - eta) * s0 + eta * s1
            s0_norm_square = np.sum(s0 ** 2, axis=1)
            ws0 = np.sum(w * s0, axis=1)

    mean_half_width2 = np.mean(ws0)
    mean_argmax_norm2 = np.mean(s0_norm_square)

    effective_dimension = np.mean(ws0 ** 2 / s0_norm_square)
    effective_dimension2 = N_NEURONS * np.mean(ws0 / (w_norm * np.sqrt(s0_norm_square))) ** 2

    alphac_hat = 1.0 / np.mean(1.0 / theory_alpha0_cached(ws0 + kappa) / (1 + s0_norm_square))

    return (mean_half_width1, mean_argmax_norm1,
            mean_half_width2, mean_argmax_norm2,
            effective_dimension, effective_dimension2,
            alphac_hat)
