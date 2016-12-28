# coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def matching_pursuit(A, y, sparsity_level, **kwargs):
    """ Matching pursuit algorithm. """
    tol = kwargs.get('stopping_tolerance',
                     0)  # Not ultimatively necessary if sparsity_level is given
    start_time = timer()
    residual = y
    support = []
    solution = np.zeros(A.shape[1])
    d_norm = np.square(np.linalg.norm(A, axis=0))
    while len(support) < sparsity_level or np.linalg.norm(residual) <= tol:
        projected_residual = A.T.dot(residual)
        new_index = np.argmax(np.abs(projected_residual))
        residual = residual - projected_residual[new_index] / d_norm[new_index] * \
            (A[:, new_index])
        solution[new_index] += projected_residual[new_index] / d_norm[new_index]
        support.append(new_index)
        support = list(set(support))  # Uniquify support
    elapsed_time = timer() - start_time
    return solution, elapsed_time, sorted(support)
