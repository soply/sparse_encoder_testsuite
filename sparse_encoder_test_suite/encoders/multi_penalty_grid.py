#coding: utf8
from timeit import default_timer as timer

import numpy as np
from lasso import lasso
from lar import least_angle_regression

def mp_lasso_grid(A, y, real_support, sparsity_level, beta_min, beta_max,
                  n_beta, beta_scaling, **kwargs):
    """ Performs multi-penalty lasso path algorithm with a grid search on the
    regularization parameter beta. The multi-penalty functional is given by

        min 1/2 ||A(u+v) - y||_2^2 + alpha ||u||_1 + beta/2 ||v||_2^2.

    For each fixed beta, this problem can be translated into an ordinary lasso
    problem such that we can use the Lasso-path algorithm to reach solution
    the solution path. This is done for specific beta's, defined as a grid of
    the given input parameters.
    """
    return aux_mp_grid(A, y, real_support, sparsity_level, beta_min, beta_max,
                    n_beta, beta_scaling, method = 'lasso')

def mp_lars_grid(A, y, real_support, sparsity_level, beta_min, beta_max,
                  n_beta, beta_scaling, **kwargs):
    """ Performs multi-penalty least angle regression algorithm with a grid
    search on the regularization parameter beta. The multi-penalty functional
    is given by

        min 1/2 ||A(u+v) - y||_2^2 + alpha ||u||_1 + beta/2 ||v||_2^2.

    For each fixed beta, this problem can be translated into an ordinary lasso
    problem such that we can use the LAR algorithm to reach solution
    the solution path. This is done for specific beta's, defined as a grid of
    the given input parameters.
    """
    return aux_mp_grid(A, y, real_support, sparsity_level, beta_min, beta_max,
                    n_beta, beta_scaling, method = 'lar')

def aux_mp_grid(A, y, real_support, sparsity_level, beta_min, beta_max,
                n_beta, beta_scaling, method, **kwargs):
    """ Auxiliary method for mp_lasso_grid and mp_lars_grid since both calls are
    almost equal. This is called from both methods to reduce/avoid code
    duplication. """
    suppress_warning =  kwargs.get('suppress_warning', False)
    start = timer()
    # SVD of A
    U, S, V = np.linalg.svd(A.dot(A.T))
    max_iter = kwargs.get('max_iter', 2 * sparsity_level + 10)
    if beta_scaling == 'linscale':
        beta_range = np.linspace(beta_min, beta_max, n_beta)
    elif beta_scaling == 'logscale':
        beta_range = np.logspace(beta_min, beta_max, n_beta)
    results = []
    for beta in beta_range:
        B_beta, y_beta = calc_B_y_beta(A, y, U, S, beta)
        # Calculate LAR path
        if method == 'lar':
            coefs, elapsed_time, support = least_angle_regression(B_beta, y_beta,
                                                                  sparsity_level)
        elif method == 'lasso':
            # Calculate Lasso path
            coefs, elapsed_time, support = lasso(B_beta, y_beta, real_support,
                                                 sparsity_level)
        else:
            raise RuntimeError("Method must be 'lar' or 'lasso'.")
        if np.array_equal(support, real_support):
            break
    # If not break occured, support and coefs from last run are taken as default
    elapsed_time = timer() - start
    return coefs, elapsed_time, support

def calc_B_y_beta(A, y, svdAAt_U, svdAAt_S, beta):
    """ Auxiliary function calculating the matrix B_beta and vector y_beta
    given by

            B_beta = (Id + A*A.T/beta)^(-1/2) * A,
            y_beta = (Id + A*A.T/beta)^(-1/2) * y.

    For speed-up, we rely on reusing the SVD of AAt that has been calculated in
    the initialisation. The modification through beta and the exponent is
    realised by manipulating the SVDs and readjusting the matrix.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        The sensing matrix A in the problem 1/2 || Ax - y ||_2^2 + alpha ||x||_1.

    y : array, shape (n_measurements)
        The vector of measurements

    svdAAt_U : array, shape (n_measurements, n_measurements)
        Matrix U of the singular value decomposition of A*A^T.

    svdAAt_S : array, shape (n_measurements)
        Array S with singular values of singular value decomposition of A*A^T.

    beta : Positive real number
        Parameter beta

    Returns
    ----------
    Tuple (B_beta, y_beta) calculated from input data.
    """
    tmp = svdAAt_U.dot(np.diag(np.sqrt(beta / (beta + svdAAt_S)))).dot(svdAAt_U.T)
    return tmp.dot(A), tmp.dot(y)
