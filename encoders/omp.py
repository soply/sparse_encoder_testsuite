# coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def orthogonal_matching_pursuit(A, y, sparsity_level, **kwargs):
    """ Orthogonal matching pursuit wrapper for scipy. """
    start_time = timer()
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity_level)
    omp.fit(A, y)
    elapsed_time = timer() - start_time
    coefs = omp.coef_
    support = coefs.nonzero()[0]
    return coefs, elapsed_time, support
