"""
Check for an input data-set of vectors X and labeling y if they are
linearly separable by a plane (with bias).

This method finds the max-margin linear separation either using primal
or dual formalisms.

Author: Uri Cohen (MATLAB original)
Converted to Python with cvxpy replacing CPLEX.
"""

import warnings
import numpy as np
import cvxpy as cp


def assert_warn(condition, message):
    """Print a warning if condition is False (mirrors MATLAB assert_warn)."""
    if not condition:
        print(f"Warning: {message}")


def check_linear_separability_svm(X, y, tolerance=1e-10, solve_dual=False, max_iterations=0):
    """
    Check linear separability via SVM quadratic programming.

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Data matrix. N = dimensionality, M = number of samples.
    y : ndarray, shape (1, M) or (M,)
        Labels, must be +1/-1.
    tolerance : float
        Feasibility / optimality tolerance.
    solve_dual : bool
        If True, solve the dual formulation; otherwise solve the primal.
    max_iterations : int
        Maximum iterations (0 = default/unlimited).

    Returns
    -------
    separable : bool
    best_w : ndarray, shape (N+1, 1)
    margin : float
    flag : int
    nsv : int
    sv_indices : ndarray
    lagrange_multipliers : ndarray
    """
    N, M = X.shape
    assert X.shape == (N, M), 'X must be NxM'

    y = np.asarray(y, dtype=float).ravel()
    assert y.shape == (M,), 'y must be M labels'
    assert np.all(np.abs(y) == 1), 'y must be +1/-1'

    # Remove empty samples: MATLAB all(isfinite(X),1) checks along dim 1
    # (along rows), producing a (1, M) mask -- True if entire column is finite.
    J = np.all(np.isfinite(X), axis=0)  # shape (M,)
    X = X[:, J]
    y = y[J]
    M = int(np.sum(J))

    # Default values for failure
    best_w = np.full((N + 1, 1), np.nan)
    separable = False
    margin = np.nan
    flag = 0
    nsv = 0
    sv_indices = np.array([], dtype=int)
    lagrange_multipliers = np.array([])

    if solve_dual:
        # Dual formulation
        # MATLAB: min 0.5*a'*H*a + f'*a  s.t. y'*a = 0, a >= 0
        # where H = (X.*y)' * (X.*y), f = -ones(1,M)
        Xy = X * y[np.newaxis, :]  # (N, M)
        H = Xy.T @ Xy  # (M, M)
        f = -np.ones(M)

        a_var = cp.Variable(M)
        objective = cp.Minimize(0.5 * cp.quad_form(a_var, cp.psd_wrap(H)) + f @ a_var)
        constraints = [
            y @ a_var == 0,
            a_var >= 0,
        ]
        prob = cp.Problem(objective, constraints)

        solver_kwargs = {}
        if max_iterations > 0:
            solver_kwargs['max_iter'] = max_iterations

        try:
            prob.solve(solver=cp.OSQP, eps_abs=tolerance, eps_rel=tolerance, **solver_kwargs)
        except cp.SolverError:
            try:
                prob.solve(solver=cp.SCS, eps=tolerance, **solver_kwargs)
            except cp.SolverError:
                flag = -100
                return separable, best_w, margin, flag, nsv, sv_indices, lagrange_multipliers

        # Map solver status to MATLAB-like flag
        if prob.status in ('infeasible', 'infeasible_inaccurate'):
            flag = -1
        elif prob.status in ('unbounded', 'unbounded_inaccurate'):
            flag = -2
        elif prob.status == 'optimal':
            flag = 1
        elif prob.status == 'optimal_inaccurate':
            flag = 5
        else:
            # Unknown status
            flag = 0

        if flag >= 0 and a_var.value is not None:
            a = a_var.value.copy()
            L = -prob.value  # MATLAB: L = -L

            assert_warn(L >= 0, f'L={L:.1f} flag={flag}')
            assert np.all(a >= -tolerance), 'Negative kkt coefficients found'
            bias_cond = abs(y @ a)
            assert bias_cond < 1e-1, f'Bias condition does not hold: {bias_cond:.1e} (flag={flag})'

            # sv_indices: MATLAB find(a > max(a)*1e-3) returns 1-based; Python 0-based
            sv_indices = np.where(a > np.max(a) * 1e-3)[0]
            lagrange_multipliers = a
            nsv = len(sv_indices)
            w = Xy @ a  # (N,)

            Xw = X.T @ w  # (M,)
            # MATLAB: b = mean(Xw(sv_indices) - y(sv_indices)')
            b = np.mean(Xw[sv_indices] - y[sv_indices])
            best_w = np.concatenate([w, [-b]])[:, np.newaxis]  # (N+1, 1)

            Xb = np.vstack([X, np.ones((1, M))])  # (N+1, M)
            separable = bool(np.all(np.sign(best_w.T @ Xb) == y))

            assert_warn(separable, f'The solution of the dual problem is not separable (flag={flag})')
    else:
        # Primal formulation
        # MATLAB: min 0.5*w'*H*w + f'*w  s.t. -Xy'*w <= -1
        # where H = eye(N+1) with H(N+1,N+1)=0, f = zeros(N+1)
        H = np.eye(N + 1)
        H[N, N] = 0.0
        f = np.zeros(N + 1)

        Xy = np.vstack([X, np.ones((1, M))]) * y[np.newaxis, :]  # (N+1, M)
        Aineq = -Xy.T  # (M, N+1)
        bineq = -np.ones(M)

        w_var = cp.Variable(N + 1)
        objective = cp.Minimize(0.5 * cp.quad_form(w_var, cp.psd_wrap(H)) + f @ w_var)
        constraints = [
            Aineq @ w_var <= bineq,
        ]
        prob = cp.Problem(objective, constraints)

        solver_kwargs = {}
        if max_iterations > 0:
            solver_kwargs['max_iter'] = max_iterations

        try:
            try:
                prob.solve(solver=cp.OSQP, eps_abs=tolerance, eps_rel=tolerance, **solver_kwargs)
            except cp.SolverError:
                prob.solve(solver=cp.SCS, eps=tolerance, **solver_kwargs)
        except (cp.SolverError, Exception) as e:
            flag = -100
            msg = str(e)
            if '1256: Basis singular.' in msg:
                print('Warning: Basis singular')
            else:
                warnings.warn(f'solver failed: {msg}')
            return separable, best_w, margin, flag, nsv, sv_indices, lagrange_multipliers

        # Map solver status to MATLAB-like flag
        if prob.status in ('infeasible', 'infeasible_inaccurate'):
            flag = -1
        elif prob.status in ('unbounded', 'unbounded_inaccurate'):
            flag = -2
        elif prob.status == 'optimal':
            flag = 1
        elif prob.status == 'optimal_inaccurate':
            flag = 5
        else:
            flag = 0

        w = w_var.value
        if w is not None and np.all(np.isfinite(w)):
            L = prob.value
            assert_warn(flag <= 0 or L >= 0, f'Got a negative result L={L:.1f} (flag={flag})')
            Xw = Xy.T @ w  # (M,)
            assert_warn(
                flag < 0 or flag == 5 or np.all(Xw - 1 >= -1e-4),
                f'Violation of kkt conditions: {np.min(Xw - 1):.1e} (flag={flag}, |w|={np.linalg.norm(w):.1e})'
            )
            # sv_indices: MATLAB find returns 1-based; Python 0-based
            sv_indices = np.where((Xw - 1) < 1e-3)[0]
            nsv = len(sv_indices)
            best_w = w[:, np.newaxis]  # (N+1, 1)

    # Final checks (common to both paths)
    if best_w is None or not np.all(np.isfinite(best_w)):
        flag = -1
        best_w = np.full((N + 1, 1), np.nan)
        return separable, best_w, margin, flag, nsv, sv_indices, lagrange_multipliers

    assert best_w.shape == (N + 1, 1)
    assert np.all(np.isfinite(best_w))

    Xb = np.vstack([X, np.ones((1, M))])  # (N+1, M)
    predictions = best_w.T @ Xb  # (1, M)
    assert not np.all(np.sign(predictions) == -y), 'Sign problem, reversed w'

    separable = bool(np.all(np.sign(predictions) == y))
    wnorm = np.linalg.norm(best_w[:N], 2)
    if wnorm == 0:
        margin = np.nan
    else:
        margin = float(np.min(predictions * y) / wnorm)

    assert_warn(
        separable or np.isnan(margin) or margin < 0,
        f'Inseparable with positive margin={margin:.1e} flag={flag}(numeric issue)'
    )
    assert not separable or np.isnan(margin) or margin >= 0

    return separable, best_w, margin, flag, nsv, sv_indices, lagrange_multipliers
