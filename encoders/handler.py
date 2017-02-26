#coding: utf8
from timeit import default_timer as timer

import numpy as np

from calculation_utilities.general import relative_error as calc_relative_error
from encoders.cosamp import cosamp
from encoders.iht import iterative_hard_thresholding
from encoders.lar import least_angle_regression
from encoders.lasso import lasso
from encoders.mp import matching_pursuit
from encoders.omp import orthogonal_matching_pursuit
from encoders.romp import regularized_orthogonal_matching_pursuit
from encoders.sp import subspace_pursuit

__list_of_encoders__ = ["mp", "omp", "lar", "lasso", "iht", "romp", "cosamp",
                  "sp", "pmp", "pomp", "plar", "plasso", "piht", "promp",
                  "pcosamp", "psp"]

def recover_support(A, y, u_real, v_real, method, sparsity_level, verbose=True):
    """ Recover the support of u_real given the sampling operator A and the
    measurements y. Method specifies one of the methods

    -lars: Least angle regression
    -lasso: Lasso-path algorithm
    -omp: Orthogonal matching pursuit algorithm

    which is then filtered for the best support of the given sparsity level. For
    lars and omp, there is only 1 support to each sparsity level that is a
    candidate to approximate the support of u_real, but for lars there can be
    more.

    Implementations used from scikit-learn. """
    target_support = np.where(u_real)[0]
    if method == "lar":
        result = least_angle_regression(A, y, sparsity_level)
    elif method == "lasso":
        result = lasso(A, y, target_support, sparsity_level)
    elif method == "mp":
        result = matching_pursuit(A, y, sparsity_level)
    elif method == "omp":
        result = orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "iht":
        result = iterative_hard_thresholding(A, y, sparsity_level, verbose=True)
    elif method == "romp":
        result = regularized_orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "cosamp":
        result = cosamp(A, y, sparsity_level)
    elif method == "sp":
        result = subspace_pursuit(A, y, sparsity_level)
    elif method in ["pmp", "pomp", "plar", "plasso", "plar", "piht", "promp",
                    "pcosamp", "psp"]:
        # Calculate preconditioned system
        start_time = timer()
        V_lp, y_new = get_preconditioned_system(A, y)
        # Call method again without p in method name
        result = recover_support(V_lp, y_new, u_real, v_real, method[1:],
                                 sparsity_level, verbose=verbose)
        elapsed_time = start_time - timer()  #  Replace time with total time
        return result[0], result[1], result[2], elapsed_time, result[4]
    else:
        raise RuntimeError("Encoder not found. Use one of {0}.".format(
                                                        __list_of_encoders__))
    # Postprocess solution
    coefs, elapsed_time, support = result
    relative_error = calc_relative_error(coefs, u_real)
    success = np.array_equal(support, target_support)
    if verbose:
        print "Finished example with method {0} in {1} seconds.".format(method,
                                                                        elapsed_time)
        print "Recovered: {0}   Correct: {1}".format(support, target_support)
        print "Success: {0}".format(success)
        print "Error: {0}".format(relative_error)
    return success, support, target_support, elapsed_time, relative_error


def get_preconditioned_system(A, y):
    U, S, V = np.linalg.svd(A)
    y_new = np.diag(1.0 / S).dot(U.T).dot(y)
    Pi = np.zeros(A.shape)
    np.fill_diagonal(Pi, 1.0)
    V_lp = Pi.dot(V)
    return V_lp, y_new


def check_method_validity(method, verbose = False):
    if method not in __list_of_encoders__:
        print("Method not found. Use one of {0}.\n".format(__list_of_encoders__))
        return False
    else:
        return True
