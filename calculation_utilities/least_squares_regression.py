#coding: utf8
import numpy as np
from pykrylov.lls import LSMRFramework

def least_squares_regression(matrix, rhs, support = None):
    """ Method performs a least squares regression for the problem
        ||matrix[:,support] * x[support] - rhs||_2
    by calculating the least squares solution x. The output will have the
    size of the matrix second dimension, although only values on the indices
    given by the support are nonzero. """
    return regularised_least_squares_regression(0, matrix, rhs, support)

def regularised_least_squares_regression(reg_param, matrix, rhs, support = None):
    """ Method performs a regularised least squares regression, i.e. solves
        || matrix[:,support] * x[support] - rhs ||_2^2 + reg_param * ||x[support]||_2^2 -> min
    for the given data. If no data is given, this method assumes that the
    stored matrix A, as well as the stored right hand side y shall be used.

    The implementation is based on the Golub-Kahan bidiagonalization process
    from the pykrylov package. This is necessary especially for lage scale
    matrices. """
    if support is None:
        support = np.arange(matrix.shape[1]) # Full support
    sol = np.zeros(matrix.shape[1])
    lsmr_solver = LSMRFramework(matrix[:,support])
    lsmr_solver.solve(rhs, damp = reg_param)
    sol[support] = lsmr_solver.x
    return sol
