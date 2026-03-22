import warnings
import time
import numpy as np
from scipy.integrate import quad


def theory_alpha0(kappa):
    """Computes theoretical capacity alpha_0(kappa) via numerical integration.

    Translated from theory_alpha0.m
    """
    if np.isnan(kappa):
        return np.nan
    if kappa > 100:
        return kappa ** -2
    Dt = lambda t: np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
    f = lambda t: Dt(t) * (t + kappa) ** 2
    I, _ = quad(f, -kappa, np.inf)
    a = I ** -1
    return a


# Module-level cache (analogous to MATLAB globals)
_CACHED_THEORY_ALPHA0 = None
_CACHED_THEORY_ALPHA0_KAPPAS = None


def theory_alpha0_cached(kappa):
    """Cached version of theory_alpha0. Pre-computes a lookup table on first call
    and uses linear interpolation for subsequent calls.

    Translated from theory_alpha0_cached.m
    """
    global _CACHED_THEORY_ALPHA0, _CACHED_THEORY_ALPHA0_KAPPAS

    if _CACHED_THEORY_ALPHA0 is None or _CACHED_THEORY_ALPHA0_KAPPAS is None:
        T = time.time()
        # MATLAB: -50:0.01:100
        _CACHED_THEORY_ALPHA0_KAPPAS = np.linspace(-50, 100, 15001)
        _CACHED_THEORY_ALPHA0 = np.zeros(_CACHED_THEORY_ALPHA0_KAPPAS.shape)
        for i in range(len(_CACHED_THEORY_ALPHA0_KAPPAS)):
            _CACHED_THEORY_ALPHA0[i] = theory_alpha0(_CACHED_THEORY_ALPHA0_KAPPAS[i])
        print('Created alpha0 cache (took {:.1f} sec)'.format(time.time() - T))

    kappa_arr = np.asarray(kappa, dtype=float)
    original_shape = kappa_arr.shape
    kappa = np.atleast_1d(kappa_arr)

    # MATLAB: I = kappa>100;
    I = kappa > 100
    alpha = np.full(kappa.shape, np.nan)
    # MATLAB: alpha(I) = kappa(I).^-2;
    alpha[I] = kappa[I] ** -2
    # MATLAB: alpha(~I) = interp1(..., 'linear', inf)
    # np.interp uses left/right for out-of-bounds (interp1 extrapval=inf)
    alpha[~I] = np.interp(kappa[~I], _CACHED_THEORY_ALPHA0_KAPPAS, _CACHED_THEORY_ALPHA0,
                           left=np.inf, right=np.inf)

    # MATLAB: assert_warn(all(isfinite(alpha(:)) | isnan(alpha(:))), ...)
    if not np.all(np.isfinite(alpha.ravel()) | np.isnan(alpha.ravel())):
        warnings.warn('Infinite values. Kappa range [{:.3e}, {:.3e}]'.format(
            np.min(kappa.ravel()), np.max(kappa.ravel())))
    # MATLAB: assert_warn(all(alpha(:)>0 | isnan(alpha(:))), ...)
    if not np.all((alpha.ravel() > 0) | np.isnan(alpha.ravel())):
        warnings.warn('Negative values. Kappa range [{:.3e}, {:.3e}]'.format(
            np.min(kappa.ravel()), np.max(kappa.ravel())))

    return alpha.reshape(original_shape)
