import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr

from scipy.optimize._lsq.common import (
    find_active_constraints, make_strictly_feasible, solve_lsq_trust_region,
    solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
    right_multiplied_operator, regularized_lsq_operator,
    CL_scaling_vector, compute_grad, compute_jac_scale,
    update_tr_radius, scale_for_robust_loss_function, print_header_nonlinear,
    print_iteration_nonlinear
)

from scipy.optimize._lsq.trf import select_step
from scipy.optimize import OptimizeResult


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """
    Check termination condition for nonlinear least squares.
    Custom monkeypatch implementation that fixes xtol check
    """
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (1 + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None


def trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev,
               x_scale, loss_function, tr_solver, tr_options, verbose):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    f_augmented = np.zeros((m + n))
    if tr_solver == 'exact':
        J_augmented = np.empty((m + n, n))
    elif tr_solver == 'lsmr':
        reg_term = 0.0
        regularize = tr_options.pop('regularize', True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        # Now compute variables in "hat" space. Here, we also account for
        # scaling introduced by `x_scale` parameter. This part is a bit tricky,
        # you have to write down the formulas and see how the trust-region
        # problem is formulated when the two types of scaling are applied.
        # The idea is that first we apply `x_scale` and then apply Coleman-Li
        # approach in the new variables.

        # v is recomputed in the variables after applying `x_scale`, note that
        # components which were identically 1 not affected.
        v[dv != 0] *= scale_inv[dv != 0]

        # Here, we apply two types of scaling.
        d = v**0.5 * scale

        # C = diag(g * scale) Jv
        diag_h = g * dv * scale

        # After all this has been done, we continue normally.

        # "hat" gradient.
        g_h = d * g

        f_augmented[:m] = f
        if tr_solver == 'exact':
            J_augmented[:m] = J * d
            J_h = J_augmented[:m]  # Memory view.
            J_augmented[m:] = np.diag(diag_h**0.5)
            U, s, V = svd(J_augmented, full_matrices=False)
            V = V.T
            uf = U.T.dot(f_augmented)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h, diag=diag_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            lsmr_op = regularized_lsq_operator(J_h, (diag_h + reg_term)**0.5)
            gn_h = lsmr(lsmr_op, f_augmented, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode='economic')
            JS = J_h.dot(S)  # LinearOperator does dot too.
            B_S = np.dot(JS.T, JS) + np.dot(S.T * diag_h, S)
            g_S = S.T.dot(g_h)

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                p_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                p_h = S.dot(p_S)

            p = d * p_h  # Trust-region solution in the original space.
            step, step_h, predicted_reduction = select_step(
                x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)
