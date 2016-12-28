#coding: utf8
from timeit import default_timer as timer

import numpy as np

def iterative_hard_thresholding(A, y, sparsity_level, **kwargs):
    """
    Iterative hard thresholding algorithm to find approximative solutions to the
    problem:
        min 1/2*||Ax - y||_2^2 + alpha ||x||_0

    Returns the solution as a 1D numpy array (also if y is given as a
    2D numpy array).
    """
    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-6)
    verbose = kwargs.get('verbose', False)

    starttime = timer()
    x_old = np.ones(A.shape[1])
    x_new = np.zeros(A.shape[1])
    Aty = A.T.dot(y)
    # Scale linear system
    scaling_factor = np.linalg.norm(A, ord=2)
    A = 1.0/scaling_factor * A
    y = 1.0/scaling_factor * y
    relative_error = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_new)
    iteration = 0
    while relative_error > tol and iteration < max_iter:
        iteration += 1
        x_old = x_new
        x_new = x_old + Aty - A.T.dot(A).dot(x_old)
        ind = np.argpartition(np.abs(x_new), -sparsity_level)[-sparsity_level:]
        mask = np.zeros(A.shape[1]).astype("bool")
        mask[ind] = True
        x_new[~mask] = 0.0
        relative_error = np.linalg.norm(x_new - x_old)/np.linalg.norm(x_new)
        if verbose:
            print "Iteration {0}:    Support = {1}      Entries = {2}     Relative_error = {3}".format(
            iteration, ind, x_new[ind], relative_error)
    elapsed_time = timer() - starttime
    return x_new, elapsed_time, np.where(x_new)[0]
