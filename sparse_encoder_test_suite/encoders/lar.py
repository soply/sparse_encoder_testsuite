# coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import lars_path


def least_angle_regression(A, y, sparsity_level, **kwargs):
    """ Least angle regression wrapper for scipy. """
    start = timer()
    alphas, active, coefs = lars_path(A, y, max_iter=sparsity_level,
                                      method="lar", return_path=True)
    elapsed_time = timer() - start
    coefs = coefs[:, -1]
    return coefs, elapsed_time, sorted(active)
