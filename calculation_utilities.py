#coding: utf8
import numpy as np
from pykrylov.lls import LSMRFramework

def calc_relative_error(u1, u2):
    """ Doc"""
    assert(u1.shape == u2.shape)
    return np.linalg.norm(u1 - u2) / np.linalg.norm(u2)

def mutual_incoherence(A):
    """ Calculates the mutual incoherence
            u = max_(i != j) |<Ai, Aj>|
    where Ai and Aj are the i-th and j-th column of A. """
    M, N = A.shape
    mutual_incoherence = 0
    for i in range(N):
        for j in range(i+1,N):
            tmp_mi = np.abs(A[:,i].dot(A[:,j]))
            if tmp_mi > mutual_incoherence:
                mutual_incoherence = tmp_mi
    return tmp_mi

def symmetric_support_difference(support1, support2):
    """ Adds indices that belong to either the support of u_calc but not the
    support of u_real, or vice versa to the support of u_real but not the
    support of u_calc to a list.

    Returns the list of non-overlapping indices as well as the length of
    non-overlapping nodes. """
    return len(np.setdiff1d(support1, support2)) + \
        len(np.setdiff1d(support2, support1))

def least_squares_regression(matrix, rhs, support = None):
    """ Method performs a least squares regression for the problem
        ||matrix[:,support] * x[support] - rhs||_2
    by calculating the least squares solution x. The output will have the
    size of the matrix second dimension, although only values on the indices
    given by the support are nonzero. """
    return regularised_least_squares_regression(0,matrix, rhs, support)

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
