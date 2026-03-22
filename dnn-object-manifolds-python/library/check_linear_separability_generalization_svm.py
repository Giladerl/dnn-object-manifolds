"""
Check for an input data-set of vectors X and labeling y if they are
linearly separable by a plane (with bias).

X is (N, M, P) and represents M samples from P clusters, each of size N.
y is (1, P) or (P,) and represents the cluster labeling.

This method finds the max-margin linear separation using either primal
or dual formalism.

To handle generalization with large problems, a small number of
conditions (indicated by 'initial_conditions') is used to find a
separating plane, and then the sample with the worst field is added,
iterating until all points are classified correctly or the problem
is unseparable (no slack variables are used here).

Note this implementation adds the worst item per object, not globally.
Thus samples_used is samples used per object and max_samples is max
samples per object (unlike the version with slack variables).

Author: Uri Cohen (MATLAB original)
Converted to Python with cvxpy replacing CPLEX.
"""

import numpy as np

from library.sample_indices import sample_indices
from library.check_linear_separability_svm import check_linear_separability_svm


def check_linear_separability_generalization_svm(
    Xs, y, tolerance=1e-10, solve_dual=False, max_iterations=0,
    max_samples=None, initial_conditions=None, verbose=False, rng=None
):
    """
    Wraps check_linear_separability_svm with iterative sample addition
    for generalization testing when max_samples > 0.

    Parameters
    ----------
    Xs : ndarray, shape (N, M, P)
        Data matrix. N = dimensionality, M = samples per cluster,
        P = number of clusters.
    y : ndarray, shape (1, P) or (P,)
        Labels, must be +1/-1.
    tolerance : float
        Feasibility / optimality tolerance.
    solve_dual : bool
        If True, solve the dual formulation; otherwise solve the primal.
    max_iterations : int
        Maximum iterations (0 = default/unlimited).
    max_samples : int or None
        Maximum samples per object to use. None defaults to M.
    initial_conditions : int or None
        Number of initial samples per cluster. None defaults to min(5, M).
    verbose : bool
        Print progress messages.
    rng : numpy.random.Generator or None
        Random number generator for sample_indices.

    Returns
    -------
    separable : bool
    best_w : ndarray, shape (N+1, 1)
    margin : float
    samples_used : int
    nsv : int
    sv_indices : ndarray
    lagrange_multipliers : ndarray
    """
    # Assert input dimensions
    assert Xs.ndim == 3, 'Xs must be NxMxP'
    N, M, P = Xs.shape
    assert Xs.shape == (N, M, P), 'Xs must be NxMxP'

    y = np.asarray(y, dtype=float).ravel()
    assert y.shape == (P,), 'y must be P labels'
    assert np.all(np.abs(y) == 1), 'y must be +1/-1'

    a = np.all(np.isfinite(Xs), axis=0)  # (M, P)
    assert np.all(a), 'Objects with empty samples are not currently supported'

    if max_samples is None:
        max_samples = M
    if initial_conditions is None:
        initial_conditions = min(5, M)

    # Data with bias
    Xb = np.concatenate([Xs, np.ones((1, M, P))], axis=0)  # (N+1, M, P)
    assert Xb.shape == (N + 1, M, P), 'Xb must be (N+1)xMxP'

    # Find indices of initial conditions
    # MATLAB: sample_indices(M, initial_conditions, P) returns (P, K)
    # Python: sample_indices(M, initial_conditions, P, rng) returns (P, K)
    I = sample_indices(M, initial_conditions, P, rng=rng)  # (P, K)
    assert I.shape == (P, initial_conditions)

    # Look for separability with current indices until data is not
    # separable or problem limits are encountered.
    separable = True
    test_margin = 0.0
    train_margin = 1.0
    K = I.shape[1]

    # Initialize outputs
    best_w = np.full((N + 1, 1), np.nan)
    margin = np.nan
    samples_used = 0
    nsv = 0
    sv_indices = np.array([], dtype=int)
    lagrange_multipliers = np.array([])

    while separable and test_margin < 0.99 * train_margin and K <= max_samples:
        currXs = np.zeros((N, K, P))
        for i in range(P):
            currXs[:, :, i] = Xs[:, I[i, :], i]

        currX = currXs.reshape((N, K * P), order='F')  # reshape columns: MATLAB reshape is column-major
        # MATLAB: currY = reshape(repmat(y, [K, 1]), [1, K*P])
        # repmat(y, [K,1]) tiles y (1,P) into (K,P), then reshape to (1, K*P)
        # column-major reshape means: for each cluster, K copies of that label
        currY = np.tile(y[np.newaxis, :], (K, 1)).reshape((1, K * P), order='F')

        samples_used = K

        (separable, best_w, train_margin, _flag, nsv, sv_indices,
         lagrange_multipliers_samples) = check_linear_separability_svm(
            currX, currY, tolerance, solve_dual, max_iterations
        )

        if lagrange_multipliers_samples.size == 0:
            lagrange_multipliers = np.array([])
        else:
            # MATLAB: reshape(lagrange_multipliers_samples, [K, P])
            lagrange_multipliers_samples = lagrange_multipliers_samples.reshape(
                (K, P), order='F'
            )
            lagrange_multipliers = np.zeros((M, P))
            for i in range(P):
                lagrange_multipliers[I[i, :], i] = lagrange_multipliers_samples[:, i]

        if separable:
            # Expand I by one column
            I = np.concatenate([I, np.zeros((P, 1), dtype=int)], axis=1)
            K = I.shape[1]

            wnorm = np.linalg.norm(best_w[:N], 2)
            test_margin = train_margin
            # Add the worst sample of each cluster
            for i in range(P):
                # MATLAB: h = y(i)*best_w'*Xb(:,:,i)
                h = y[i] * (best_w.T @ Xb[:, :, i])  # (1, M)
                h_norm = h.ravel() / wnorm  # (M,)
                i0 = np.argmin(h_norm)
                h0 = h_norm[i0]
                test_margin = min(test_margin, h0)
                I[i, K - 1] = i0  # K-1 because 0-based (last column)

            margin = test_margin
            if verbose:
                print(f'    separable using {K - 1} samples per cluster: '
                      f'training margin={train_margin:.1e} generalization={test_margin:.1e}')
        else:
            margin = train_margin
            if verbose:
                print(f'    inseparable using {K} samples per cluster: '
                      f'training margin={train_margin:.1e}')

    separable = margin > 0

    return separable, best_w, margin, samples_used, nsv, sv_indices, lagrange_multipliers
