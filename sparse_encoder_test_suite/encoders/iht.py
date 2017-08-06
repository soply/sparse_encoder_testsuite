#coding: utf8
from timeit import default_timer as timer

import numpy as np

from bp import basis_pursuit


def iterative_hard_thresholding(A, y, sparsity_level, x0 = None, **kwargs):
    """
    Iterative hard thresholding algorithm to find approximative solutions to the
    problem:
        min 1/2*||Ax - y||_2^2 + alpha ||x||_0

    Returns the solution as a 1D numpy array (also if y is given as a
    2D numpy array).
    """
    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-4)
    # If support is the same for so many times in a row, we stop the iteration
    tol_n_equal_supports = kwargs.get('n_equal_supports', 20)
    verbose = True #kwargs.get('verbose', False)

    starttime = timer()
    x_old = np.ones(A.shape[1])
    if x0 is None:
        x_new = np.zeros(A.shape[1])
    else:
        x_new = x0
    Aty = A.T.dot(y)
    # Scale linear system
    scaling_factor = np.linalg.norm(A, ord=2)
    A = 1.0/scaling_factor * A
    y = 1.0/scaling_factor * y
    relative_error = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_new)
    iteration = 0
    current_support = np.where(x_new)[0]
    n_equal_supports = 0
    while relative_error > tol and iteration < max_iter and \
            n_equal_supports < tol_n_equal_supports:
        iteration += 1
        x_old = x_new
        x_new = x_old + Aty - A.T.dot(A).dot(x_old)
        ind = np.argpartition(np.abs(x_new), -sparsity_level)[-sparsity_level:]
        mask = np.zeros(A.shape[1]).astype("bool")
        mask[ind] = True
        x_new[~mask] = 0.0
        relative_error = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_new)
        support = np.where(mask)[0]
        if np.array_equal(current_support, support):
            n_equal_supports += 1
        else:
            current_support = support
            n_equal_supports = 0
        if verbose:
            print "Iteration {0}:    Support = {1}      Entries = {2}     Relative_error = {3}".format(
            iteration, sorted(ind), x_new[ind], relative_error)
    elapsed_time = timer() - starttime
    return x_new, elapsed_time, np.where(x_new)[0]

def l1_iterative_hard_thresholding(A, y, sparsity_level, **kwargs):
    """
    Iterative hard thresholding with basis pursuit (l1) warmup  to find
    approximative solutions to the problem:

        min 1/2*||Ax - y||_2^2 + alpha ||x||_0

    In this version, we start IHT from a solution of the BP problem, i.e. we
    have a warm-up step to find a good initialization.
    Returns the solution as a 1D numpy array (also if y is given as a
    2D numpy array).
    """
    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-4)
    verbose = kwargs.get('verbose', False)
    starttime = timer()
    x_new, bp_time, bp_support = basis_pursuit(A, y, abs_tol = tol,
                                            rel_tol = tol, max_iter = max_iter)
    x, iht_time, support = iterative_hard_thresholding(A, y, sparsity_level,
                                            x0 = x_new, **kwargs)
    elapsed_time = timer() - starttime
    return x, elapsed_time, support
