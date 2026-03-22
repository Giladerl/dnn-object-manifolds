"""
Binary search for the critical neuron count Nc where ~50% of random binary
dichotomies are separable.

Translated from check_binary_dichotomies_capacity2.m
"""

import math
import time
import numpy as np
from .check_binary_dichotomies_sampled_features import check_binary_dichotomies_sampled_features


def check_binary_dichotomies_capacity(
    XsAll, EXPECTED_PRECISION=0.05, verbose=False, random_labeling_type=0,
    precision=1, max_samples=0, global_preprocessing=0, features_type=0,
    jumps=None, rng=None
):
    """
    Calculate the capacity: the minimal N for which over half of binary
    dichotomies are achievable.  Works in jumps to find an upper bound,
    then does binary search.

    This version uses an adaptive number of samples and does not collect
    manifold properties to minimize computational cost.

    Parameters
    ----------
    XsAll : np.ndarray, shape (N_NEURONS, N_SAMPLES, N_OBJECTS)
        Full tuning function data.
    EXPECTED_PRECISION : float
        Target precision for separability estimation.
    verbose : bool or int
        Verbosity level.
    random_labeling_type : int
        0=IID, 1=balanced, 2=sparse.
    precision : int
        Binary search precision (stop when range <= precision).
    max_samples : int
        Max samples per object (0 = use all).
    global_preprocessing : int
        Preprocessing mode.
    features_type : int
        0=sub-sample, 1=first n (PCA), 2=random projections.
    jumps : int or None
        Step size for initial sweep. Auto-determined if None.
    rng : np.random.Generator or None

    Returns
    -------
    Nc : int or float
        Critical number of neurons (np.nan if not found).
    separability_results : np.ndarray, shape (N_NEURONS,)
        Separability fraction at each neuron count (NaN where not tested).
    Ns : np.ndarray
        1-based neuron counts where results are available.
    n_neuron_samples_used : np.ndarray, shape (N_NEURONS,)
        Number of neuron samples used at each tested point.
    n_support_vectors : np.ndarray, shape (N_NEURONS, N_OBJECTS)
        Support vector counts per object at each tested point.
    """
    # Get problem dimensions
    assert XsAll.ndim == 3, 'Data must be [N_NEURONS, N_SAMPLES, N_OBJECTS]'
    N_NEURONS, N_SAMPLES, N_OBJECTS = XsAll.shape

    # Determine jump size
    if jumps is None:
        if N_NEURONS < 100:
            jumps = 10
        elif N_NEURONS < 1024:
            if N_NEURONS % 100 == 0:
                jumps = 100
            else:
                jumps = 128
        else:
            if N_NEURONS % 500 == 0:
                jumps = 500
            else:
                jumps = 512

    if verbose:
        print(f' {N_NEURONS} neurons {N_SAMPLES} conditions {N_OBJECTS} objects')

    # Result variables (indexed 0..N_NEURONS-1; index k corresponds to
    # neuron count k+1, i.e. MATLAB's 1-based index n maps to Python index n-1)
    separability_results = np.full(N_NEURONS, np.nan)
    n_neuron_samples_used = np.full(N_NEURONS, np.nan)
    n_support_vectors = np.full((N_NEURONS, N_OBJECTS), np.nan)

    def _eval_at(n):
        """Evaluate separability at neuron count n (1-based). Stores results."""
        idx = n - 1  # 0-based array index
        if np.isnan(separability_results[idx]):
            separability, nsv_per_object = check_binary_dichotomies_sampled_features(
                XsAll, n, EXPECTED_PRECISION,
                random_labeling_type, max_samples, global_preprocessing,
                verbose=(verbose > 1) if isinstance(verbose, (int, float)) else False,
                features_type=features_type, rng=rng,
            )
            n_neuron_samples_used[idx] = np.sum(np.isfinite(separability))
            separability_results[idx] = np.nanmean(separability)
            n_support_vectors[idx, :] = nsv_per_object

    # ------------------------------------------------------------------
    # Phase 1: Find max N in jumps
    # MATLAB: J = unique([jumps:jumps:(N_NEURONS-mod(N_NEURONS,jumps)), N_NEURONS])
    # ------------------------------------------------------------------
    top = N_NEURONS - (N_NEURONS % jumps)
    J_list = list(range(jumps, top + 1, jumps))
    J_list.append(N_NEURONS)
    J = sorted(set(J_list))

    n = J[-1]  # default: last element if loop doesn't break
    for n_j in J:
        if verbose:
            print(f'  initial check at n={n_j}')
            T = time.time()

        _eval_at(n_j)

        if verbose:
            idx = n_j - 1
            print(f'  N={n_j} {separability_results[idx]:.2f} separable '
                  f'({int(n_neuron_samples_used[idx])} samples, '
                  f'took {time.time() - T:.1f} sec)')

        if separability_results[n_j - 1] == 1:
            if verbose:
                print('  Found initial max N')
            n = n_j
            break

    # ------------------------------------------------------------------
    # Phase 2: Binary search until precision is met
    # ------------------------------------------------------------------
    max_N = n  # 1-based neuron count from phase 1

    # MATLAB: min_N = find(separablity_results == 0, 1, 'last')
    zero_indices = np.where(separability_results == 0)[0]
    if len(zero_indices) > 0:
        min_N = int(zero_indices[-1]) + 1  # convert 0-based index to 1-based neuron count
    else:
        min_N = 0

    # Compute min/max over non-NaN entries (MATLAB's min/max skip NaN)
    def _min_val_idx():
        """Return (min_value, min_index_1based) ignoring NaN."""
        tmp = separability_results.copy()
        tmp[np.isnan(tmp)] = np.inf
        idx0 = int(np.argmin(tmp))
        return tmp[idx0], idx0 + 1

    def _max_val_idx():
        """Return (max_value, max_index_1based) ignoring NaN."""
        tmp = separability_results.copy()
        tmp[np.isnan(tmp)] = -np.inf
        idx0 = int(np.argmax(tmp))
        return tmp[idx0], idx0 + 1

    min_value, min_index = _min_val_idx()
    max_value, max_index = _max_val_idx()

    n1 = np.nan
    n2 = np.nan

    while True:
        cond1 = (max_N - min_N > precision)
        cond2 = (min_value > 0 and (min_index - 1) > precision
                 and math.ceil((min_index + 1) / 2) != n)
        cond3 = (max_value < 1 and (N_NEURONS - max_index) > precision
                 and math.ceil((max_index + N_NEURONS) / 2) != n)

        if not (cond1 or cond2 or cond3):
            break

        if cond1:
            n = math.ceil((max_N + min_N) / 2)
        elif cond2:
            n = math.ceil((min_index + 1) / 2)
        elif cond3:
            n = math.ceil((max_index + N_NEURONS) / 2)
        else:
            break

        if n == n1 or n == n2:
            break

        if verbose:
            print(f'  looking at n={n} [{min_N}-{max_N}] '
                  f'({int(max_N - min_N > precision)} '
                  f'{int(np.nanmin(separability_results) > 0)} '
                  f'{int(np.nanmax(separability_results) < 1)})')
            T = time.time()

        _eval_at(n)

        if verbose:
            idx = n - 1
            print(f'  N={n} {separability_results[idx]:.2f} separable '
                  f'({int(n_neuron_samples_used[idx])} samples, '
                  f'took {time.time() - T:.1f} sec)')

        min_value, min_index = _min_val_idx()
        max_value, max_index = _max_val_idx()

        if max_N - min_N > precision:
            if separability_results[n - 1] > 0.5:
                max_N = n
            else:
                min_N = n

        n2 = n1
        n1 = n

    # ------------------------------------------------------------------
    # Compute final capacity
    # ------------------------------------------------------------------
    # MATLAB: Nc = find(separablity_results >= 0.5, 1)  — first 1-based index
    ge_half = np.where(separability_results >= 0.5)[0]
    if len(ge_half) > 0:
        Nc = int(ge_half[0]) + 1  # 1-based neuron count
        if verbose:
            print(f' Critical N={Nc} alpha={N_OBJECTS / Nc:.2f}')
    else:
        Nc = np.nan
        if verbose:
            print(' Critical N not found')

    # MATLAB: Ns = find(isfinite(separablity_results))  — returns 1-based indices
    Ns = np.where(np.isfinite(separability_results))[0] + 1

    return Nc, separability_results, Ns, n_neuron_samples_used, n_support_vectors
