#coding: utf8
import numpy as np
from pykrylov.lls import LSMRFramework


def least_squares_regression(matrix, rhs, support = None):
    """ Method performs a least squares regression for the problem

    ||matrix[:,support] * x[support] - rhs||_2^2 -> min (1)

    The output will have the size of the matrix second dimension, although only
    values on the indices given by the support can be nonzero.

    Parameters
    -------------
    matrix : array, shape (n_measurements, n_features)
        Matrix in the problem (1).

    rhs : array, shape (n_measurements)
        Measurements in the problem (1).

    support, optional : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution is allowed to be nonzero. If not specified, the
        regression is performed on all entries.

    Returns
    --------------
    Numpy array of shape (n_features) that is the solution to the unpenalised
    least squares regression problem given above. Is only non-zero in entries
    specified by the support.

    Remarks
    -------------
    Uses the LSMRFramework with Golub-Kahan bidiagonalization process to solve
    the least squares problem. If the solution is not unique (underdetermined)
    system, consult the docs of LSMR to see which solution is provided.
    """
    return regularised_least_squares_regression(0.0, matrix, rhs, support)

def regularised_least_squares_regression(reg_param, matrix, rhs, support = None):
    """ Method performs a regularised least squares regression, i.e. solves

    ||matrix[:,support]*x[support]-rhs||_2^2+beta*||x[support]||_2^2 -> min  (1)

    The output will have the size of the matrix second dimension, although only
    values at indices given by the support can be nonzero.

    Parameters
    -------------
    reg_param : Positive, real number
        Regularisation parameter in the problem (1).

    matrix : array, shape (n_measurements, n_features)
        Matrix in the problem (1).

    rhs : array, shape (n_measurements)
        Measurements in the problem (1).

    support : array, shape (n_support)
        Python list or numpy array with integer corresponding to the support on
        which the solution is allowed to be nonzero. If not specified, the
        regression is performed on all entries.

    Returns
    --------------
    Numpy array of shape (n_features) that is the solution to the unpenalised
    least squares regression problem given above. Is only non-zero in entries
    specified by the support.

    Remarks
    -------------
    Uses the LSMRFramework with Golub-Kahan bidiagonalization process to solve
    the least squares problem. If the solution is not unique (ie. if reg_param=0)
    system, consult the docs of LSMR to see which solution is provided.
    The implementation is based on the Golub-Kahan bidiagonalization process
    from the pykrylov package. """
    sol = np.zeros(matrix.shape[1])
    lsmr_solver = LSMRFramework(matrix[:, support])
    lsmr_solver.solve(rhs, damp=reg_param)
    sol[support] = lsmr_solver.x
    return sol
