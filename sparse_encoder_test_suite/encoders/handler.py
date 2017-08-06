#coding: utf8
""" Handler and some utility methods for support recovery. Main interface to
    communicate with the different methods for support recovery."""

__author__ = "Timo Klock"

from timeit import default_timer as timer

import numpy as np

from ..calculation_utilities.general import relative_error as calc_relative_error
from cosamp import cosamp
from iht import iterative_hard_thresholding, l1_iterative_hard_thresholding
from lar import least_angle_regression
from lasso import lasso
from mp import matching_pursuit
from bp import basis_pursuit
from omp import orthogonal_matching_pursuit
from romp import regularized_orthogonal_matching_pursuit
from sp import subspace_pursuit
from multi_penalty_grid import mp_lasso_grid, mp_lars_grid
from elastic_net import elastic_net

# Available methods (romp, promp do not work currently)
__list_of_encoders__ = ["mp", "omp", "lar", "lasso", "iht", "l1iht", "romp",
                        "cosamp", "sp", "bp", "pmp", "pomp", "plar", "plasso",
                        "piht", "pl1iht", "promp", "pcosamp", "psp",
                        "mp_lasso_grid", "mp_lars_grid", "enet", "penet", 'pbp']

def recover_support(A, y, u_real, method, sparsity_level, verbose=True,
                    **kwargs):
    """ Handler method to call the different sparse encoders. Ultimatively uses
    encoder specified under method with the given data and recovers the support.
    Returns a success flag, the final support, the target support, the elapsed
    time and the relative error to the real solution.

    Parameters
    --------------
    A : np.array, shape (n_measurements, n_features)
        Sampling matrix

    y : np.array, shape (n_measurements)
        Vector of measurements.

    u_real : np.array, shape (n_features)
        Signal that generated the measurements y under sampling of A, ie.
        A * u_real = y (+ potential signal/measurements noise).

    method : python string
        Sparse encoder that should be used. Should be in the
        __list_of_encoders__ specified above. promp and romp do not work
        currently.

    sparsity_level : python Integer
        Support size of the generating signal u_real.

    verbose : python Boolean
        Controls print-outs of this method.

    kwargs : python dictionary
        Keyword arguments than will be passed to the solver if necessary.

    Returns
    -----------------
    success : True if correct support was recovered, false otherwise
    support : Support recovered at the end.
    target_support : Support of u_real.
    elapsed_time : time spent for calculating the support/solution.
    relative_error : relative l2 error between the recovered solution and u_real.
    """
    target_support = np.where(u_real)[0]
    if method == "lar":
        result = least_angle_regression(A, y, sparsity_level, **kwargs)
    elif method == "lasso":
        result = lasso(A, y, target_support, sparsity_level)
    elif method == 'mp_lasso_grid':
        result = mp_lasso_grid(A, y, target_support, sparsity_level, **kwargs)
    elif method == 'mp_lars_grid':
        result = mp_lars_grid(A, y, target_support, sparsity_level, **kwargs)
    elif method == 'bp':
        result = basis_pursuit(A, y, verbose = verbose, **kwargs)
    elif method == "mp":
        result = matching_pursuit(A, y, sparsity_level)
    elif method == "omp":
        result = orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "iht":
        result = iterative_hard_thresholding(A, y, sparsity_level,
                                             verbose = verbose)
    elif method == "l1iht":
        result = l1_iterative_hard_thresholding(A, y, sparsity_level,
                                                verbose = verbose)
    elif method == "romp":
        result = regularized_orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "cosamp":
        result = cosamp(A, y, sparsity_level)
    elif method == "sp":
        result = subspace_pursuit(A, y, sparsity_level)
    elif method == "enet":
        result = elastic_net(A, y, target_support, sparsity_level, **kwargs)
    elif method in ["pmp", "pomp", "plar", "plasso", "plar", "piht", "promp",
                    "pcosamp", "psp", "penet"]:
        # Calculate preconditioned system
        start_time = timer()
        V_lp, y_new = get_preconditioned_system(A, y)
        # Call method again without p in method name
        result = recover_support(V_lp, y_new, u_real, method[1:],
                                 sparsity_level, verbose=verbose, **kwargs)
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


def get_preconditioned_system(A, y, S_tol = 1e-13):
    """ Transforms given data into a preconditioned system by the Puffer
    transformation. The Puffer transformation is given as F = U D^dagger U^T where
    A = UDV^T is the SVD of A and D^dagger is equal to D^-1 if dii > 0 and 0
    else. The transformed data is given by FA, Fy (we will skip the leading U,
    since it doesn't influence the solutions but saves computational time)

    Parameters
    -------------
    A : np.array, shape (n_measurements, n_features)
        Sampling matrix

    y : np.array, shape (n_measurements)
        Vector of measurements.

    S_tol : float
        Singular values below this value will be set to zero, not inverted.

    Returns
    -------------
    Returns precoditioned data, that is FA and Fy according to the formula above.

    Sources
    -------------
    [1] Jia, Jinzhu, and Karl Rohe. "Preconditioning to comply with the
        irrepresentable condition." arXiv preprint arXiv:1208.5584 (2012).
    """
    U, S, V = np.linalg.svd(A, full_matrices = False)
    new_A = np.linalg.pinv(A)
    S[S < S_tol] = 0.0
    S[S > 0] = 1.0/S[S > 0]
    y_new = np.diag(S).dot(U.T).dot(y)
    S[S > 0] = 1.0
    A_new = np.diag(S).dot(V)
    return A_new, y_new


def check_method_validity(method, verbose = False):
    """ Checks if the given method is valid, ie. available.

    Parameters
    ------------
    method : python string
        Method name to check

    verbose : Boolean
        True if list of available methods should be shown in case of failure,
        false otherwise.

    Returns
    -------------
    True if the given method is available, false otherwise.
    """
    if method not in __list_of_encoders__:
        if verbose:
            print("Method not found. Use one of {0}.\n".format(__list_of_encoders__))
        return False
    else:
        return True
