#coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import lars_path

def lasso(A, y, real_support, sparsity_level, **kwargs):
    """ Lasso path wrapper for scipy. """
    suppress_warning =  kwargs.get('suppress_warning', False)
    max_iter = kwargs.get('max_iter', 2 * sparsity_level + 10)
    start = timer()
    alphas, active, coefs = lars_path(A, y, max_iter = max_iter,
        method = "lasso", return_path = True)
    binary_coefs = coefs.astype("bool")
    potential_candidates = np.where(np.sum(binary_coefs, 0) == sparsity_level)[0]
    for potential_supp_cand in potential_candidates:
        # Check if correct support is under the given supports.
        if np.array_equal(binary_coefs[:,potential_supp_cand], real_support):
            support = np.where(binary_coefs[:,potential_supp_cand])[0]
            coefs = coefs[:,potential_supp_cand]
            break
    else:
        # Correct support has not been found. Take first entry default sol.
        try:
            support = np.where(binary_coefs[:,potential_candidates[0]])[0]
            coefs = coefs[:, potential_candidates[0]]
        except IndexError:
            # Case potential_candidates is empty
            support = np.zeros(sparsity_level)
            coefs = np.zeros(sparsity_level)
    elapsed_time = timer() - start
    return coefs, elapsed_time, support
