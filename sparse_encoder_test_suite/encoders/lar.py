# coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import lars_path


def least_angle_regression(A, y, sparsity_level, **kwargs):
    """ Least angle regression wrapper for scipy. """
    suppress_warning = kwargs.get('suppress_warning', False)
    start = timer()
    alphas, active, coefs = lars_path(A, y, max_iter=sparsity_level,
                                      method="lar", return_path=True)
    elapsed_time = timer() - start
    if len(active) != sparsity_level and not suppress_warning:
        active = np.zeros(sparsity_level)
        print ('''least_angle_regression Warning: The number of active indices
               {0} do not equal desired sparsity level {1}. Using empty support
               as default solution.'''.format(len(active), sparsity_level))
    coefs = coefs[:, -1]
    return coefs, elapsed_time, sorted(active)
