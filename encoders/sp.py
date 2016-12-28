# coding: utf8
from timeit import default_timer as timer

import numpy as np


def subspace_pursuit(A, y, sparsity_level, **kwargs):
    """ Subspace pursuit algorithm. """
    verbose = kwargs.get('verbose', True)
    max_iter = kwargs.get('max_iter', 1000)  # FIXME: Figure out good default val.
    tol = kwargs.get('tol', 1e-15)  # FIXME: Figure out good default val.
    tol_consec_error = kwargs.get('tol_ce', 1e-15)

    start_time = timer()
    support = np.argsort(np.abs(A.T.dot(y)))[-sparsity_level:]
    residual = y - A[:, support].dot(np.linalg.pinv(A[:, support])).dot(y)
    iterations = 0
    new_error = np.linalg.norm(residual)
    while iterations < max_iter:
        next_active_indices = np.argsort(np.abs(A.T.dot(residual)))[-sparsity_level:]
        tentative_support = np.union1d(support, next_active_indices)
        sol_on_support = np.linalg.pinv(A[:, tentative_support]).dot(y)
        indx_wrt_tentative = np.argsort(np.abs(sol_on_support))[-sparsity_level:]
        support_new = tentative_support[indx_wrt_tentative]
        residual_new = y - A[:, support_new].dot(np.linalg.pinv(A[:, support_new])).dot(y)
        # Update errors and if terminated
        old_error = new_error
        new_error = np.linalg.norm(residual_new)
        if new_error > old_error or new_error < tol or \
                np.abs(new_error - old_error) / (new_error) < tol_consec_error:
            support_new = support
            break
        else:
            support = support_new
            residual = residual_new
            iterations += 1
    # Calculate signal entries via least squares regression
    sol_on_support = np.linalg.pinv(A[:, support]).dot(y)
    full_sol = np.zeros(A.shape[1])
    full_sol[support] = sol_on_support
    elapsed_time = timer() - start_time
    return full_sol, elapsed_time, sorted(support)
