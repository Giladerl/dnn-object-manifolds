"""
Cost function for squared off-diagonal correlations (used in Stiefel optimization).

Translated from square_corrcoeff_full_cost.m
"""

import numpy as np


def square_corrcoeff_full_cost(V, X):
    """
    Compute the sum of squared off-diagonal normalized correlation coefficients
    after removing the low-rank component V, and its gradient w.r.t. V.

    Parameters
    ----------
    V : np.ndarray, shape (N, K)
        Orthonormal basis vectors.
    X : np.ndarray, shape (P, N)
        Data matrix (P manifolds, N features).

    Returns
    -------
    cost : float
    gradient : np.ndarray, shape (N, K)
    """
    P, N = X.shape
    K = V.shape[1]
    assert V.shape == (N, K)

    # Calculate cost
    C = X @ X.T                                   # (P, P)
    c = X @ V                                     # (P, K)
    c0 = np.diag(C) - np.sum(c ** 2, axis=1)     # (P,)
    Fmn = (C - c @ c.T) ** 2 / (c0[:, None] * c0[None, :])  # (P, P)
    cost = float(np.sum(Fmn) / 2.0)

    # Calculate gradient
    # MATLAB reshapes:
    # X1 = reshape(X, [1, P, N])
    # X2 = reshape(X, [P, 1, N])
    # C1 = reshape(c, [P, 1, 1, K])
    # C2 = reshape(c, [1, P, 1, K])
    X1 = X.reshape(1, P, N)          # (1, P, N)
    X2 = X.reshape(P, 1, N)          # (P, 1, N)
    C1 = c.reshape(P, 1, 1, K)       # (P, 1, 1, K)
    C2 = c.reshape(1, P, 1, K)       # (1, P, 1, K)

    # Precompute 2D terms
    ratio = (C - c @ c.T) / (c0[:, None] * c0[None, :])      # (P, P)
    ratio_sq = (C - c @ c.T) ** 2 / (c0[:, None] * c0[None, :]) ** 2  # (P, P)

    # Gmni has effective shape (P, P, N, K) via broadcasting.
    # Each term uses ratio or ratio_sq broadcast from (P, P) -> (P, P, 1, 1).

    # Term 1: -ratio .* (C1 .* X1)
    #   C1: (P, 1, 1, K), X1: (1, P, N) -> (1, P, N, 1)
    #   C1 * X1[..., None]: (P, P, N, K)
    #   ratio[:, :, None, None]: (P, P, 1, 1)
    Gmni = -ratio[:, :, None, None] * (C1 * X1[:, :, :, None])

    # Term 2: -ratio .* (C2 .* X2)
    #   C2: (1, P, 1, K), X2: (P, 1, N) -> (P, 1, N, 1)
    #   C2 * X2[..., None]: (P, P, N, K)
    Gmni = Gmni - ratio[:, :, None, None] * (C2 * X2[:, :, :, None])

    # Term 3: +ratio_sq .* (c0 .* C2 .* X1)
    #   MATLAB: bsxfun(@times, c0, bsxfun(@times, C2, X1))
    #   c0 is (P,) column in MATLAB -> broadcasts along first axis
    #   C2: (1, P, 1, K), X1: (1, P, N) -> (1, P, N, 1)
    #   C2 * X1[..., None]: (1, P, N, K)
    #   c0[:, None, None, None]: (P, 1, 1, 1) broadcasts against (1, P, N, K) -> (P, P, N, K)
    Gmni = Gmni + ratio_sq[:, :, None, None] * (c0[:, None, None, None] * (C2 * X1[:, :, :, None]))

    # Term 4: +ratio_sq .* (c0' .* C1 .* X2)
    #   MATLAB: bsxfun(@times, c0', bsxfun(@times, C1, X2))
    #   c0' is (1, P) row in MATLAB -> broadcasts along second axis
    #   C1: (P, 1, 1, K), X2: (P, 1, N) -> (P, 1, N, 1)
    #   C1 * X2[..., None]: (P, 1, N, K)
    #   c0[None, :, None, None]: (1, P, 1, 1) broadcasts against (P, 1, N, K) -> (P, P, N, K)
    Gmni = Gmni + ratio_sq[:, :, None, None] * (c0[None, :, None, None] * (C1 * X2[:, :, :, None]))

    # gradient = reshape(sum(sum(Gmni, 1), 2), [N, K])
    # MATLAB sum(Gmni, 1) sums over first dim (m), sum(..., 2) sums over second dim (n)
    # In Python axis 0 is m, axis 1 is n
    gradient = np.sum(np.sum(Gmni, axis=0), axis=0)  # (N, K)

    return cost, gradient
