# coding: utf8
from timeit import default_timer as timer

import numpy as np


def basis_pursuit(A, y, **kwargs):
    """
    The conventional basis pursuit (BP) program solves the optimization problem

        argmin_x ||x||_1 s.t. Ax = y

    i.e. without relaxing the constraints about data fidelity. In this
    implementation we solve this convex program using an ADMM approach.
    """
    max_iter = kwargs.get('max_iter', 1000)
    abs_tol = kwargs.get('tol', 1e-4)
    rel_tol = kwargs.get('tol', 1e-4)
    verbose = kwargs.get('verbose', False)
    alpha = kwargs.get('alpha', 1.2) # Over relxation parameter
    rho = kwargs.get('rho', 1.0) # Augmented Lagrangian parameter.
    def shrink(vec, kappa):
        return np.maximum(np.zeros(vec.shape[0]), vec-kappa) - \
                    np.maximum(np.zeros(vec.shape[0]), -vec-kappa);
    starttime = timer()
    # Problem sizes
    m, n = A.shape
    # Containers
    x = np.zeros(n);
    z = np.zeros(n);
    u = np.zeros(n);
    # Precomputations
    AAt = A.dot(A.T)
    P = np.identity(n) - A.T.dot(np.linalg.lstsq(AAt, A)[0]) # P = eye(n) - A' * (AAt \ A);
    q = A.T.dot(np.linalg.lstsq(AAt, y)[0])
    # Iteration counter
    n_iter = 0
    # ADMM Iteration with (alpha, rho)
    while n_iter < max_iter:
        # Updating x
        x = P.dot(z - u) + q
        # Updating z with relaxation
        zold = z
        x_hat = alpha * x + (1.0 - alpha) * zold;
        z = shrink(x_hat + u, 1.0/rho)
        u = u + (x_hat - z)
        obj_val = np.sum(np.abs(x))
        primal_error = np.linalg.norm(x - z)
        dual_error = np.linalg.norm(-rho * (z-zold))
        eps_pri = np.sqrt(n) * abs_tol + \
                        np.maximum(np.linalg.norm(x), np.linalg.norm(-z)) * rel_tol
        eps_dual = np.sqrt(n) * abs_tol + np.linalg.norm(rho * u) * rel_tol
        if verbose:
            print "Iter {4}  Objective {5}   Primal error: {0} (EPS: {1})  Dual error: {2} (EPS: {3})".format(
                primal_error, eps_pri, dual_error, eps_dual, n_iter, obj_val)
        if primal_error < eps_pri and dual_error < eps_dual:
            if verbose:
                print "Finished after {0} iterations".format(n_iter)
            break
        else:
            n_iter += 1
    elapsed_time = timer() - starttime
    return x, elapsed_time, np.where(z)[0]
