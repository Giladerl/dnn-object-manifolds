import numpy as np


def calc_low_dimension_manifold(tuning_function, D, data_randomization=0,
                                reduce_global_mean=True, radii_factor=1, rng=None):
    """Reduce tuning functions to D dimensions via SVD, with randomization modes
    for center structure.

    Parameters
    ----------
    tuning_function : ndarray, shape (N_NEURONS, N_SAMPLES, N_OBJECTS)
    D : int
        Number of SVD dimensions to keep.
    data_randomization : int
        Randomization mode (0-8).
    reduce_global_mean : bool
        Whether to subtract the global mean across samples and objects.
    radii_factor : float
        Factor to scale centers by (default 1, unchanged).
    rng : numpy.random.Generator or None
        Random number generator. If None, a default Generator is created.

    Returns
    -------
    result_tuning_function : ndarray, same shape as tuning_function
    """
    if rng is None:
        rng = np.random.default_rng()

    tuning_function = tuning_function.copy()
    N_NEURONS, N_SAMPLES, N_OBJECTS = tuning_function.shape

    # Reduce the global mean
    if reduce_global_mean:
        # MATLAB: bsxfun(@minus, tuning_function, mean(mean(tuning_function, 2), 3))
        # mean over samples (axis=1), then over objects (axis=2), result is (N_NEURONS, 1, 1)
        global_mean = tuning_function.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True)
        tuning_function = tuning_function - global_mean

    # Shuffle object assignment
    if data_randomization == 8:
        # MATLAB reshapes column-major: reshape(tf, [N_NEURONS, N_SAMPLES*N_OBJECTS])
        shuffled_tuning = tuning_function.reshape(N_NEURONS, N_SAMPLES * N_OBJECTS, order='F')
        sampleIndices = rng.permutation(N_SAMPLES * N_OBJECTS)
        shuffled_tuning = shuffled_tuning[:, sampleIndices]
        tuning_function = shuffled_tuning.reshape(N_NEURONS, N_SAMPLES, N_OBJECTS, order='F')

    # Calculate centers — mean over samples (axis=1), shape (N_NEURONS, 1, N_OBJECTS)
    Centers = tuning_function.mean(axis=1, keepdims=True)

    if data_randomization == 1:
        # squeeze(Centers) → (N_NEURONS, N_OBJECTS)
        Centers_sq = Centers.squeeze(axis=1)
        # ns = squeeze(sqrt(sum(Centers.^2,1)))' → norm of each object's center
        ns = np.sqrt((Centers_sq ** 2).sum(axis=0))  # (N_OBJECTS,)
        Q, _ = np.linalg.qr(Centers_sq)
        # bsxfun(@times, Q(:,1:N_OBJECTS), ns./radii_factor), reshaped to Centers shape
        newCenters = (Q[:, :N_OBJECTS] * (ns / radii_factor)[np.newaxis, :]).reshape(
            Centers.shape, order='F')
        ns2 = np.sqrt((newCenters.squeeze(axis=1) ** 2).sum(axis=0))
        assert np.max(np.abs(ns2 * radii_factor - ns)) < 1e-10

    elif data_randomization == 2 or data_randomization == 3:
        Centers_sq = Centers.squeeze(axis=1)
        ns = np.sqrt((Centers_sq ** 2).sum(axis=0))
        randCenters = rng.standard_normal(Centers.shape)
        randCenters_sq = randCenters.squeeze(axis=1)
        ns1 = np.sqrt((randCenters_sq ** 2).sum(axis=0))
        # bsxfun(@times, squeeze(randCenters), ns./ns1./radii_factor)
        newCenters = (randCenters_sq * (ns / ns1 / radii_factor)[np.newaxis, :]).reshape(
            Centers.shape, order='F')
        ns2 = np.sqrt((newCenters.squeeze(axis=1) ** 2).sum(axis=0))
        assert np.max(np.abs(ns2 * radii_factor - ns)) < 1e-8

    elif data_randomization == 4:
        Centers_sq = Centers.squeeze(axis=1)
        ns = np.sqrt((Centers_sq ** 2).sum(axis=0))
        randCenters = rng.standard_normal(Centers.shape)
        randCenters_sq = randCenters.squeeze(axis=1)
        ns1 = np.sqrt((randCenters_sq ** 2).sum(axis=0))
        # bsxfun(@times, squeeze(randCenters), mean(ns)./ns1./radii_factor)
        newCenters = (randCenters_sq * (np.mean(ns) / ns1 / radii_factor)[np.newaxis, :]).reshape(
            Centers.shape, order='F')
        ns2 = np.sqrt((newCenters.squeeze(axis=1) ** 2).sum(axis=0))
        assert np.max(np.abs(ns2 * radii_factor - np.mean(ns))) < 1e-8

    else:
        newCenters = Centers / radii_factor

    # Result variables
    result_tuning_function = np.zeros(tuning_function.shape)
    for i in range(N_OBJECTS):
        # cF = tuning_function(:,:,i) - Centers(:,:,i)
        # Centers(:,:,i) is (N_NEURONS, 1), broadcasts over N_SAMPLES
        cF = tuning_function[:, :, i] - Centers[:, :, i]

        # Remove missing samples
        J = np.all(np.isfinite(cF), axis=0)
        cF = cF[:, J]

        # Economy SVD
        U, s, Vt = np.linalg.svd(cF, full_matrices=False)
        # SD = zeros(size(S)); SD(1:D,1:D) = S(1:D,1:D);
        # dF = U*SD*V';
        # This is rank-D reconstruction: (U[:, :D] * s[:D]) @ Vt[:D, :]
        s_trunc = np.zeros_like(s)
        s_trunc[:D] = s[:D]
        dF = (U * s_trunc[np.newaxis, :]) @ Vt

        # neuronIndices = randperm(N_NEURONS) — computed every iteration
        neuronIndices = rng.permutation(N_NEURONS)

        if data_randomization == 3 or data_randomization == 4 or data_randomization == 5:
            result_tuning_function[:, :, i] = dF[neuronIndices, :] + newCenters[:, :, i]
        elif data_randomization == 7:
            result_tuning_function[:, :, i] = dF[neuronIndices, :] + newCenters[neuronIndices, :, i]
        else:
            result_tuning_function[:, :, i] = dF + newCenters[:, :, i]

    return result_tuning_function
