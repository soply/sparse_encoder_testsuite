#coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import lars_path
from ..calculation_utilities.general import find_support_minimalSD

def lasso(A, y, target_support, sparsity_level, **kwargs):
    """ Lasso path wrapper for scipy. """
    suppress_warning =  kwargs.get('suppress_warning', False)
    max_iter = kwargs.get('max_iter', 2 * sparsity_level + 10)
    start = timer()
    alphas, active, coefs = lars_path(A, y, max_iter = max_iter,
        method = "lasso", return_path = True)
    binary_coefs = coefs.astype("bool")
    support, index = find_support_minimalSD(binary_coefs, target_support)
    coefs = coefs[:, index]
    elapsed_time = timer() - start
    return coefs, elapsed_time, support
