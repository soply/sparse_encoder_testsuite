# coding: utf8
from timeit import default_timer as timer

import numpy as np


def cosamp(A, y, sparsity_level, **kwargs):
    """ FROM cosamp.m file:

    Cosamp algorithm
       Input
           sparsity_level : sparsity of u_real
           A : measurement matrix
           y: measured vector
           u_real : correct solution
       Output
           sol: Solution found by the algorithm

    Translated from Matlab implementation by David Mary and Bob L. Sturm (see
    http://media.aau.dk/null_space_pursuits/2011/08/16/cosamp.m).
    """

    max_iter = kwargs.get('max_iter', 1000)  # FIXME: Figure out good default val.
    tol = kwargs.get('tol', 1e-10)  # FIXME: Figure out good default val.
    zero_tol = kwargs.get('zero_tol', 1e-12)  # Threshold for entry zero or active
    verbose = kwargs.get('verbose', True)
    tol_consec_error = kwargs.get('tol_ce', 1e-12)

    start_time = timer()
    current_residual = y
    iteration = 1
    support_mask = np.zeros(A.shape[1]).astype("bool")
    old_error = 10e10
    new_error = np.linalg.norm(current_residual) / np.linalg.norm(y)
    while iteration < max_iter and new_error > tol and \
            np.abs(old_error - new_error) / new_error > tol_consec_error:
        old_error = np.linalg.norm(current_residual) / np.linalg.norm(y)
        yyy = np.abs(A.T.dot(current_residual))
        vals = np.sort(yyy)[::-1]
        new_potential_indices = (yyy >= vals[2 * sparsity_level - 1]) & (yyy > zero_tol)
        support_mask[new_potential_indices] = True
        bb = np.linalg.pinv(A[:, support_mask]).dot(y)
        vals = np.sort(np.abs(bb))[::-1]
        accepted_indices = (np.abs(bb) >= vals[sparsity_level - 1]) & \
            (np.abs(bb) > zero_tol)
        support_mask[np.where(support_mask)[0][~accepted_indices]] = False
        bb_reduced = bb[accepted_indices]
        sol = np.zeros(A.shape[1])
        sol[support_mask] = bb_reduced
        current_residual = y - A[:, support_mask].dot(bb_reduced)
        iteration = iteration + 1
        new_error = np.linalg.norm(current_residual) / np.linalg.norm(y)
        if verbose:
            print "Iteration {0}     New error {1}     Support {2}".format(
                iteration, new_error, np.where(support_mask)[0])
    elapsed_time = timer() - start_time
    support = np.where(sol)[0]
    return sol, elapsed_time, support
