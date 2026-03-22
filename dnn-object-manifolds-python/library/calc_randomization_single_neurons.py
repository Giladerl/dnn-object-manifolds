"""
Apply global preprocessing modes to tuning function data.

Translated from calc_randomization_single_neurons.m
"""

import numpy as np
import warnings


def _assert_warn(condition, msg=""):
    """Issue a warning instead of raising an error (mirrors MATLAB assert_warn)."""
    if not condition:
        warnings.warn(msg if msg else "Assertion warning: condition not met")


def calc_randomization_single_neurons(full_tuning_function, global_preprocessing, rng=None):
    """
    Apply one of 8 global preprocessing modes to a tuning function array.

    Parameters
    ----------
    full_tuning_function : np.ndarray, shape (N_NEURONS, N_SAMPLES, N_OBJECTS)
        The raw tuning function data.
    global_preprocessing : int
        Preprocessing mode (1-8).  Any value outside 1-8 returns data unchanged.
    rng : np.random.Generator or None
        Random number generator (unused in this function but accepted for
        API consistency).

    Returns
    -------
    tuning_function : np.ndarray
        Preprocessed tuning function data.
    """
    N_NEURONS, N_SAMPLES, N_OBJECTS = full_tuning_function.shape

    if global_preprocessing == 1:
        # Z-norm: subtract mean, divide by std, then add back scaled mean
        tuning_function = full_tuning_function.reshape(
            (N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_mean = np.nanmean(tuning_function, axis=1, keepdims=True)
        tuning_function = tuning_function - tuning_mean
        a = np.nanstd(tuning_function, axis=1, ddof=1, keepdims=True)
        a[a == 0] = 1
        tuning_function = tuning_function / a
        tuning_mean = tuning_mean / a
        _assert_warn(
            np.linalg.norm(
                np.nanmean(tuning_function ** 2, axis=1)
                - (N_SAMPLES * N_OBJECTS - 1) / (N_SAMPLES * N_OBJECTS),
                np.inf
            ) < 1e-10
        )
        tuning_function = tuning_function + tuning_mean
        tuning_function = tuning_function.reshape(
            (N_NEURONS, N_SAMPLES, N_OBJECTS), order='F')

    elif global_preprocessing == 2:
        # Whitening
        tuning_function = full_tuning_function.reshape(
            (N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = tuning_function - np.nanmean(tuning_function, axis=1, keepdims=True)
        pU, pS_vec, _ = np.linalg.svd(tuning_function, full_matrices=False)
        # pU: (N_NEURONS, min_dim), pS_vec: (min_dim,)
        # where min_dim = min(N_NEURONS, N_SAMPLES*N_OBJECTS)
        # MATLAB: pS = diag(diag(pS)) is a no-op (pS already diagonal)
        # Safe inversion: zero singular values stay zero (matches MATLAB backslash
        # behaviour for singular diagonal matrices).
        inv_pS = np.zeros_like(pS_vec)
        nonzero = pS_vec != 0
        inv_pS[nonzero] = 1.0 / pS_vec[nonzero]
        # pS \ pU' * tuning_function  =>  diag(inv_pS) @ pU.T @ tuning_function
        tuning_function = (inv_pS[:, np.newaxis] * (pU.T @ tuning_function)) * np.sqrt(
            N_SAMPLES * N_OBJECTS - 1)
        assert np.linalg.norm(
            np.nanmean(tuning_function ** 2, axis=1)
            - (N_SAMPLES * N_OBJECTS - 1) / (N_SAMPLES * N_OBJECTS),
            np.inf
        ) < 1e-10
        assert np.linalg.norm(
            tuning_function @ tuning_function.T / (N_SAMPLES * N_OBJECTS - 1)
            - np.eye(N_NEURONS),
            np.inf
        ) < 1e-6
        tuning_function = tuning_function.reshape(
            (N_NEURONS, N_SAMPLES, N_OBJECTS), order='F')

    elif global_preprocessing == 3:
        # Project into a subspace where the centers are decorrelated
        Xc = np.nanmean(full_tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        _, cS_vec, cVt = np.linalg.svd(Xc, full_matrices=False)
        cV = cVt.T
        cS = np.diag(cS_vec)
        I = np.eye(N_NEURONS)
        XtX = Xc.T @ Xc
        # W = (I - Xc * inv(Xc'*Xc) * Xc') + Xc * cV * inv(cS^3) * cV' * Xc'
        W = (I - Xc @ np.linalg.solve(XtX, Xc.T)) + \
            Xc @ cV @ np.linalg.matrix_power(cS, -3) @ cV.T @ Xc.T
        assert not np.any(np.isnan(W))
        assert W.shape == (N_NEURONS, N_NEURONS)
        flat = full_tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = (W @ flat).reshape(
            (N_NEURONS, N_SAMPLES, N_OBJECTS), order='F')
        decorelated_centers = np.nanmean(tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        Cc = decorelated_centers.T @ decorelated_centers
        deviation = np.linalg.norm(Cc - np.eye(N_OBJECTS), np.inf)
        _assert_warn(deviation < 1e-8,
                     'Deviation from I: {:.3e}'.format(deviation))

    elif global_preprocessing == 4 or global_preprocessing == 5:
        # Variance-preserving center decorrelation
        Xc = np.nanmean(full_tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        cU, cS_vec, cVt = np.linalg.svd(Xc, full_matrices=False)
        cV = cVt.T
        cS = np.diag(cS_vec)
        XtX = Xc.T @ Xc
        Var = np.diag(np.diag(XtX))
        I = np.eye(N_NEURONS)
        Var_sqrt = np.diag(np.sqrt(np.diag(Var)))
        # W = (I - Xc * inv(Xc'*Xc) * Xc') + cU * Var^0.5 * cV * inv(cS^2) * cV' * Xc'
        W = (I - Xc @ np.linalg.solve(XtX, Xc.T)) + \
            cU @ Var_sqrt @ cV @ np.linalg.matrix_power(cS, -2) @ cV.T @ Xc.T
        assert not np.any(np.isnan(W))
        assert W.shape == (N_NEURONS, N_NEURONS)
        flat = full_tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = (W @ flat).reshape(
            (N_NEURONS, N_SAMPLES, N_OBJECTS), order='F')
        decorelated_centers = np.nanmean(tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        Cc = decorelated_centers.T @ decorelated_centers
        deviation = np.linalg.norm(Cc - Var, np.inf)
        _assert_warn(deviation < 1e-6,
                     'Deviation from I: {:.3e}'.format(deviation))
        if global_preprocessing == 5:
            # MATLAB: global_mean = nanmean(reshape(tf, [N_NEURONS, N_SAMPLES*N_OBJECTS]), 2)
            # Result is (N_NEURONS, 1).  bsxfun(@minus, tf_3d, global_mean) broadcasts
            # (N_NEURONS,1) against (N_NEURONS, N_SAMPLES, N_OBJECTS) -- MATLAB auto-expands
            # the singleton dims.  In NumPy we need shape (N_NEURONS, 1, 1).
            global_mean = np.nanmean(
                tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F'),
                axis=1)  # (N_NEURONS,)
            tuning_function = tuning_function - global_mean[:, np.newaxis, np.newaxis]

    elif global_preprocessing == 6:
        # Project into the subspace spanned by centers, with decorrelation
        Xc = np.nanmean(full_tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        cU, cS_vec, _ = np.linalg.svd(Xc, full_matrices=False)
        cS = np.diag(cS_vec)
        # reduceW = inv(cS) @ cU'  ->  (N_OBJECTS, N_NEURONS)
        reduceW = np.linalg.solve(cS, cU.T)
        assert reduceW.shape == (N_OBJECTS, N_NEURONS)
        flat = full_tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = (reduceW @ flat).reshape(
            (N_OBJECTS, N_SAMPLES, N_OBJECTS), order='F')
        reduced_centers = np.nanmean(tuning_function, axis=1)  # (N_OBJECTS, N_OBJECTS)
        deviationI = np.linalg.norm(
            reduced_centers.T @ reduced_centers - np.eye(N_OBJECTS), np.inf)
        _assert_warn(deviationI < 1e-10,
                     'Deviation from I in centers span: {:.3e}'.format(deviationI))

    elif global_preprocessing == 7:
        # Project into center subspace (no decorrelation), subtract global mean
        Xc = np.nanmean(full_tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        cU, _, _ = np.linalg.svd(Xc, full_matrices=False)
        reduceW = cU.T  # (N_OBJECTS, N_NEURONS)
        assert reduceW.shape == (N_OBJECTS, N_NEURONS)
        flat = full_tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = (reduceW @ flat).reshape(
            (N_OBJECTS, N_SAMPLES, N_OBJECTS), order='F')
        # MATLAB: global_mean = nanmean(reshape(tf, [N_OBJECTS, N_SAMPLES*N_OBJECTS]), 2)
        # Result is (N_OBJECTS, 1).  bsxfun(@minus, tf_3d, global_mean) needs (N_OBJECTS,1,1).
        global_mean = np.nanmean(
            tuning_function.reshape((N_OBJECTS, N_SAMPLES * N_OBJECTS), order='F'),
            axis=1)  # (N_OBJECTS,)
        tuning_function = tuning_function - global_mean[:, np.newaxis, np.newaxis]

    elif global_preprocessing == 8:
        # Variance-preserving projection into center subspace, subtract global mean
        Xc = np.nanmean(full_tuning_function, axis=1)  # (N_NEURONS, N_OBJECTS)
        XtX = Xc.T @ Xc
        Var = np.diag(np.diag(XtX))
        cU, cS_vec, cVt = np.linalg.svd(Xc, full_matrices=False)
        cV = cVt.T
        cS = np.diag(cS_vec)
        Var_sqrt = np.diag(np.sqrt(np.diag(Var)))
        # reduceW = Var^0.5 * cV * inv(cS) * cU'
        reduceW = Var_sqrt @ cV @ np.linalg.solve(cS, cU.T)
        assert reduceW.shape == (N_OBJECTS, N_NEURONS)
        flat = full_tuning_function.reshape((N_NEURONS, N_SAMPLES * N_OBJECTS), order='F')
        tuning_function = (reduceW @ flat).reshape(
            (N_OBJECTS, N_SAMPLES, N_OBJECTS), order='F')
        reduced_centers = np.nanmean(tuning_function, axis=1)  # (N_OBJECTS, N_OBJECTS)
        assert np.linalg.norm(
            reduced_centers.T @ reduced_centers - Var, np.inf) < 1e-10
        global_mean = np.nanmean(
            tuning_function.reshape((N_OBJECTS, N_SAMPLES * N_OBJECTS), order='F'),
            axis=1)  # (N_OBJECTS,)
        tuning_function = tuning_function - global_mean[:, np.newaxis, np.newaxis]

    else:
        tuning_function = full_tuning_function.copy()

    return tuning_function
