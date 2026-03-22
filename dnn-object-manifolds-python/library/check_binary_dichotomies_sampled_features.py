"""
Test linear separability of binary dichotomies using sub-sampled features
with adaptive sampling until target precision is met.

Translated from check_binary_dichotomies_sampled_features2.m
"""

import time
import numpy as np
from .sample_indices import sample_indices
from .sample_random_labels import sample_random_labels
from .check_linear_separability_svm import check_linear_separability_svm
from .check_linear_separability_generalization_svm import check_linear_separability_generalization_svm
from .calc_randomization_single_neurons import calc_randomization_single_neurons


def check_binary_dichotomies_sampled_features(
    XsAll, n, expected_precision=0.05, random_labeling_type=0,
    max_samples=0, global_preprocessing=0, verbose=False, features_type=0,
    rng=None
):
    """
    Test binary dichotomies using *n* sub-sampled features.

    Adaptively samples neuron subsets (or random projections) and binary
    dichotomies until the standard deviation of the mean separability
    fraction drops below *expected_precision*.

    Parameters
    ----------
    XsAll : np.ndarray, shape (N_NEURONS, N_SAMPLES, N_OBJECTS)
        Full tuning-function data.
    n : int
        Number of features (neurons) to use per trial.
    expected_precision : float
        Target precision (std of the mean) for the separability fraction.
    random_labeling_type : int
        0 = IID, 1 = balanced, 2 = sparse.
    max_samples : int
        Max samples per object for the generalization SVM path.
        0 means use all samples (plain SVM).
    global_preprocessing : int
        Preprocessing mode (see ``calc_randomization_single_neurons``).
    verbose : bool
        Print progress information.
    features_type : int
        0 = sub-sample neurons, 1 = first *n* features (e.g. PCA),
        2 = random projections.
    rng : np.random.Generator or None
        Random number generator.  If ``None``, a new default generator is
        created.

    Returns
    -------
    separability : np.ndarray, shape (N_NEURON_SAMPLES, N_DICHOTOMIES)
        Each entry is 1.0 (separable) or 0.0 (not separable), with NaN
        for trials not yet executed when early stopping fired.
    nsv_per_cluster : np.ndarray, shape (N_OBJECTS,)
        Number of support vectors per object cluster (from the last
        dichotomy tested).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Constants (matching MATLAB exactly)
    # ------------------------------------------------------------------ #
    N_DICHOTOMIES = 1
    # Minimal number of neuron samples to try
    MIN_NEURON_SAMPLES = int(np.ceil(1.0 / expected_precision))
    # Expected precision ep = sqrt(p*q/n), for p=q=0.5: n = (0.5/ep)^2
    N_NEURON_SAMPLES = int(np.ceil((0.5 / expected_precision) ** 2))

    N_NEURONS, N_SAMPLES, N_OBJECTS = XsAll.shape

    # ------------------------------------------------------------------ #
    # Prepare feature indices / random-projection flag
    # ------------------------------------------------------------------ #
    if features_type == 0:
        features_used = sample_indices(N_NEURONS, n, N_NEURON_SAMPLES, rng=rng)
    elif features_type == 1:
        assert N_NEURON_SAMPLES == 1
        # MATLAB: features_used = zeros(1, n); features_used(1:n) = 1:n;
        features_used = np.arange(n).reshape(1, -1)  # 0-based
    else:
        assert features_type == 2
        features_used = None  # will use random projections

    max_iterations = 1000
    tolerance = 1e-10

    # ------------------------------------------------------------------ #
    # Result variables
    # ------------------------------------------------------------------ #
    current = 0
    separability = np.full((N_NEURON_SAMPLES, N_DICHOTOMIES), np.nan)
    nsv_per_cluster = np.zeros(N_OBJECTS)

    for r in range(N_NEURON_SAMPLES):
        # ----- Prepare features -----
        if features_used is None:
            # Random projections
            random_projections = rng.standard_normal((N_NEURONS, n)) / np.sqrt(N_NEURONS)
            X_flat = XsAll.reshape(N_NEURONS, N_SAMPLES * N_OBJECTS)
            X_proj = random_projections.T @ X_flat
            Xs = calc_randomization_single_neurons(
                X_proj.reshape(n, N_SAMPLES, N_OBJECTS), global_preprocessing
            )
        else:
            I = features_used[r, :]
            Xs = calc_randomization_single_neurons(XsAll[I, :, :], global_preprocessing)

        X = Xs.reshape(n, N_SAMPLES * N_OBJECTS)

        # ----- Calculate if we are above or below capacity -----
        for i in range(N_DICHOTOMIES):
            current += 1
            t_start = time.time()

            y = sample_random_labels(N_OBJECTS, random_labeling_type, rng=rng)

            if max_samples > 0:
                # Generalization SVM path: operates on 3-D Xs and per-object labels y
                (separable, _best_w, margin, samples_used,
                 _nsv, sv_indices, _lm) = check_linear_separability_generalization_svm(
                    Xs, y, tolerance=tolerance, solve_dual=False,
                    max_iterations=max_iterations, max_samples=max_samples
                )
                assert (
                    np.isnan(margin)
                    or separable == (margin > 0)
                    or (samples_used == max_samples)
                ), (
                    f"Margin and separability mismatch: {separable} {margin:.1e} "
                    f"(used {samples_used} / {max_samples} samples)"
                )
            else:
                # Plain SVM path: operates on 2-D X and expanded labels Y
                # MATLAB: Y = reshape(repmat(y, [N_SAMPLES, 1]), [1, N_SAMPLES*N_OBJECTS])
                Y = np.tile(y, (N_SAMPLES, 1)).ravel(order='F').reshape(1, -1)

                (separable, _best_w, margin, _flag,
                 _nsv, sv_indices, _lm) = check_linear_separability_svm(
                    X, Y, tolerance=tolerance, solve_dual=False,
                    max_iterations=max_iterations
                )
                assert (
                    np.isnan(margin) or separable == (margin > 0)
                ), f"Margin and separability mismatch: {separable} {margin:.1e}"

            # Map 1-D sv_indices into (N_SAMPLES, N_OBJECTS) matrix and count
            # per cluster.  MATLAB: nsv_per_cluster = zeros(N_SAMPLES, N_OBJECTS);
            # nsv_per_cluster(sv_indices) = 1;  (linear / column-major indexing)
            # nsv_per_cluster = sum(nsv_per_cluster, 1);
            nsv_mat = np.zeros((N_SAMPLES, N_OBJECTS), dtype=float)
            if sv_indices is not None and len(sv_indices) > 0:
                # MATLAB uses column-major linear indexing into an
                # (N_SAMPLES x N_OBJECTS) matrix.  In Fortran order the
                # linear index maps as:
                #   row = idx % N_SAMPLES
                #   col = idx // N_SAMPLES
                # np.unravel_index with order='F' does this correctly.
                rows, cols = np.unravel_index(sv_indices, (N_SAMPLES, N_OBJECTS), order='F')
                nsv_mat[rows, cols] = 1
            nsv_per_cluster = nsv_mat.sum(axis=0)  # sum over samples

            separability[r, i] = 1.0 if margin > 0 else 0.0

            if verbose:
                elapsed = time.time() - t_start
                print(f"   N={n} margin={margin:.1e} (took {elapsed:.1f} sec)")

        # ----- Early stopping check -----
        n_res = int(np.sum(np.isfinite(separability)))
        p = np.nansum(separability == 1) / n_res
        q = np.nansum(separability == 0) / n_res
        assert p + q == 1
        std_of_the_mean = np.sqrt(p * q / n_res)
        if n_res >= MIN_NEURON_SAMPLES and std_of_the_mean <= expected_precision:
            if verbose:
                print(f"   reached target std of the mean: {std_of_the_mean:.3f}")
            return separability, nsv_per_cluster

    return separability, nsv_per_cluster
