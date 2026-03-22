"""
Iteratively find optimal low-rank decomposition of center correlations
via Stiefel manifold optimization.

Translated from optimal_low_rank_structure2.m
"""

import time
import math
import warnings

import numpy as np

from stiefel_opt.opt_stiefel_gbb import opt_stiefel_gbb
from library.square_corrcoeff_full_cost import square_corrcoeff_full_cost


def optimal_low_rank_structure(X, MAX_K=None, verbose=1, minSquare=True,
                               N_REPEATS=1, rng=None):
    """
    Find optimal low-rank structure of center correlations.

    Parameters
    ----------
    X : np.ndarray, shape (N, P)
        Data matrix (N features, P manifolds).
    MAX_K : int or None
        Maximum rank to try. Defaults to ceil(P/2).
    verbose : int
        Verbosity level (0=silent, 1=summary, 2=detailed).
    minSquare : bool
        If True, minimize squared correlations; if False, minimize absolute.
    N_REPEATS : int
        Number of optimization repeats per rank.
    rng : np.random.Generator or None
        Random number generator. If None, creates a default one.

    Returns
    -------
    Vopt : np.ndarray or None
        Optimal basis vectors in original space, shape (N, Kopt), or None if Kopt==0.
    Xopt : np.ndarray
        Residual centers at optimal rank, shape (N, P).
    Kopt : int
        Optimal rank.
    residual_centers_norm : np.ndarray, shape (MAX_K+1, P)
        Norms of residual centers for each rank.
    mean_square_corrcoef : np.ndarray, shape (MAX_K+1,)
        Mean squared off-diagonal correlation coefficients.
    mean_abs_corrcoef : np.ndarray, shape (MAX_K+1,)
        Mean absolute off-diagonal correlation coefficients.
    mean_square_corr : np.ndarray, shape (MAX_K+1,)
        Mean squared off-diagonal correlations (unnormalized).
    mean_abs_corr : np.ndarray, shape (MAX_K+1,)
        Mean absolute off-diagonal correlations (unnormalized).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Problem dimensions
    N, P = X.shape

    # Default parameter values
    if MAX_K is None:
        MAX_K = math.ceil(P / 2)

    early_termination = True

    # Results variables
    mean_square_corr = np.full(MAX_K + 1, np.nan)
    mean_abs_corr = np.full(MAX_K + 1, np.nan)
    mean_square_corrcoef = np.full(MAX_K + 1, np.nan)
    mean_abs_corrcoef = np.full(MAX_K + 1, np.nan)
    residual_centers_norm = np.full((MAX_K + 1, P), np.nan)

    # Optimization options
    MAX_ITER = 10000
    opts = {
        'record': 0,        # no print out
        'mxitr': MAX_ITER,  # max number of iterations
        'gtol': 1e-6,       # stop control for the projected gradient
        'xtol': 1e-6,       # stop control for ||X_k - X_{k-1}||
        'ftol': 1e-8,       # stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
    }

    # Move from representation from dimension of N to P-1, the rank
    if N > P - 1:
        Q, _ = np.linalg.qr(X, mode='reduced')  # Q is (N, P)
        assert Q.shape == (N, P)
        Q = Q[:, :P - 1]  # (N, P-1)
        Xq = Q.T @ X      # (P-1, P)
        assert Xq.shape == (P - 1, P)
    else:
        Xq = X.copy()
        Q = np.eye(N)

    # Initialize cost to infinite
    bestCost = np.inf
    Vopt = None
    Xopt = None
    Kopt = 0

    V = None  # accumulates across iterations (corresponds to MATLAB V)

    for ik in range(MAX_K + 1):
        k = ik  # k goes from 0 to MAX_K
        if early_termination and (k > Kopt + 3):
            if verbose >= 1:
                print(f'Early termination Kopt={Kopt}')
            break

        t_start = time.time()

        if k == 0:
            V = None
            Xk = Xq.copy()
        else:
            best_stability = 0.0
            best_V = None
            for i_n in range(N_REPEATS):
                s = rng.standard_normal((P, 1))  # (P, 1)
                # V0 = [Xq*s, V]  -- build initial point on Stiefel manifold
                Xq_s = Xq @ s  # (P-1, 1)
                if V is None:
                    # k == 1, first iteration: V0 is just Xq*s
                    V0 = Xq_s  # (P-1, 1)
                else:
                    V0 = np.hstack([Xq_s, V])  # (P-1, k)

                # QR orthogonalize
                V0, _ = np.linalg.qr(V0, mode='reduced')
                assert V0.shape == (Q.shape[1], k)

                # Optimize on Stiefel manifold
                V1, output = opt_stiefel_gbb(V0, square_corrcoeff_full_cost,
                                             opts, Xq.T)

                if output['itr'] >= MAX_ITER:
                    warnings.warn(
                        f"Max iterations reached at k={k} ({output.get('msg', '')})")
                if not np.isfinite(output['fval']):
                    warnings.warn(
                        f"Non finite cost at k={k}: {output['fval']:.3e} "
                        f"({output.get('msg', '')})")

                # Check orthonormality
                orth_dev = np.linalg.norm(V1.T @ V1 - np.eye(k))
                assert output['itr'] >= MAX_ITER or orth_dev < 1e-6, \
                    f'Deviation from I: {orth_dev:.2e}'

                # Verify cost
                cost_after, _ = square_corrcoeff_full_cost(V1, Xq.T)
                assert (output['itr'] >= MAX_ITER
                        or not np.isfinite(output['fval'])
                        or abs(output['fval'] - cost_after) < 1e-6), \
                    f"Cost differ: {output['fval']:.3e} <> {cost_after:.3e}"

                Xk = Xq - V1 @ (V1.T @ Xq)  # (P-1, P)
                stability = np.min(
                    np.sqrt(np.sum(Xk ** 2, axis=0)) /
                    np.sqrt(np.sum(Xq ** 2, axis=0))
                )
                if stability > best_stability:
                    best_stability = stability
                    best_V = V1.copy()

                if N_REPEATS > 1 and verbose >= 2:
                    print(f' [{i_n + 1}] cost={cost_after:.3f} '
                          f'stability={stability:.3f}')

            V = best_V
            Xk = Xq - V @ (V.T @ Xq)  # (P-1, P)

        # Save output variables
        Xk_norm = np.sqrt(np.sum(Xk ** 2, axis=0))  # (P,)
        residual_centers_norm[ik, :] = Xk_norm

        Ck = Xk.T @ Xk  # (P, P)
        square_offdiagonal_corr = (Ck - np.diag(np.diag(Ck))) ** 2
        mean_square_corr[ik] = np.sum(square_offdiagonal_corr) / (P - 1) / P

        abs_offdiagonal_corr = np.abs(Ck - np.diag(np.diag(Ck)))
        mean_abs_corr[ik] = np.sum(abs_offdiagonal_corr) / (P - 1) / P

        Ck0 = (Xk.T @ Xk) / (Xk_norm[:, None] * Xk_norm[None, :])  # (P, P)
        square_offdiagonal_corr = (Ck0 - np.diag(np.diag(Ck0))) ** 2
        mean_square_offdiagonal_corr = np.sum(square_offdiagonal_corr) / (P - 1) / P
        mean_square_corrcoef[ik] = mean_square_offdiagonal_corr

        abs_offdiagonal_corr = np.abs(Ck0 - np.diag(np.diag(Ck0)))
        mean_abs_offdiagonal_corr = np.sum(abs_offdiagonal_corr) / (P - 1) / P
        mean_abs_corrcoef[ik] = mean_abs_offdiagonal_corr

        elapsed = time.time() - t_start
        if verbose >= 1:
            print(f'k={k} <square>={mean_square_offdiagonal_corr:.4f} '
                  f'<abs>={mean_abs_offdiagonal_corr:.3f} (took {elapsed:.1f} sec)')

        # Update the best results
        if minSquare:
            currentCost = mean_square_offdiagonal_corr
        else:
            currentCost = mean_abs_offdiagonal_corr

        if currentCost < bestCost:
            bestCost = currentCost
            if V is None:
                Vopt = None
            else:
                Vopt = Q @ V
            Xopt = Q @ Xk
            Kopt = k

    return (Vopt, Xopt, Kopt, residual_centers_norm,
            mean_square_corrcoef, mean_abs_corrcoef,
            mean_square_corr, mean_abs_corr)
