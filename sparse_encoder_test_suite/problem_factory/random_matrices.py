# coding: utf8
""" Methods to create different types of random matrices. These methods are
    typically called from synthetic_random_data.py to create a problem from
    random data. Types of matrices implemented:

        1) random iid Gaussian matrix
        2) Partial Random Circulant matrices with Rademacher vectors
        3)                                   with Standard Gaussian vectors
        4) Partial Random Toeplitz matrices with Rademacher vectors
        5)                                  with Standard Gaussian vectors

    More coming soon.
    """

__author__ = "Timo Klock"

import numpy as np
from scipy.linalg import circulant, toeplitz

__available_matrices__ = ['gaussian', 'prcm_rademacher', 'prcm_gaussian',
                          'prtm_rademacher', 'prtm_gaussian']

def create_sampling_matrix(n_measurements, n_features, mat_type, random_seed=None,
                           random_state=None, scaling="measurements"):
    """ Method to create a sampling matrix of size (n_measurements, n_features).
    'scaling' determines whether the columns are scaled by the number of
    measurements (making correlation measurement independent) or by
    column normalization.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    mat_type : python String
        Specifies the type of matrix that shall be constructed.

    random_seed : pthon Integer
        Random seed for numpy

    random_state : np.random_state
        Random state for numpy

    scaling: Determines scaling of the sampling matrix. Two possibilities:
        a) 'measurements': Matrix is rescaled by 1/sqrt(n_measurements),
                           making correlation independent of n_measurements.
        b) 'normalized_cols': Columns are normalized to have norm 1.

    Returns
    -----------------
    Sampling matrix as np.array of size (n_measurements, n_features).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    if mat_type == 'gaussian':
        A = _create_gaussian_matrix(n_measurements, n_features)
    elif mat_type == 'prcm_rademacher':
        A = _create_prcm_rademacher(n_measurements, n_features)
    elif mat_type == 'prcm_gaussian':
        A = _create_prcm_gaussian(n_measurements, n_features)
    elif mat_type == 'prtm_rademacher':
        A = _create_prtm_rademacher(n_measurements, n_features)
    elif mat_type == 'prtm_gaussian':
        A = _create_prtm_gaussian(n_measurements, n_features)
    else:
        raise RuntimeError(('mat_type {0} is could not be identified in matrix' +
            ' during matrix creation.\nAvailable types: {1}').format(
                mat_type, __available_matrices__))
    if scaling == "measurements":
        # Scaling to have an n_measurement invariant incoherence value
        A = 1.0 / np.sqrt(n_measurements) * A
    elif scaling == "normalized_cols":
        for i in range(A.shape[1]):
            A[:, i] = A[:, i] / np.linalg.norm(A[:, i])
    else:
        raise NotImplementedError('''Desired scaling method {0} not implmeneted.
        Please choose either of {1} or {2}.'''.format(scaling, 'measurements',
                                                      'normalized_cols'))
    return A

def _create_gaussian_matrix(n_measurements, n_features):
    """ Method to create a sampling matrix of size (n_measurements, n_features).
    'scaling' determines whether the columns are scaled by the number of
    measurements (making correlation measurement independent) or by
    column normalization.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    Returns
    -----------------
    Random Gaussian matrix as np.array of size (n_measurements, n_features).
    """
    A = np.random.randn(n_measurements, n_features)
    return A

def _create_prcm_rademacher(n_measurements, n_features):
    """ Method to create a partial random circulant matrix from a rademacher
    sequence.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    Returns
    -----------------
    Partial random circulant matrix as np.array of size (n_measurements, n_features).
    """
    rademacher_sequence = np.random.binomial(1, 0.5,
                                        size=(n_features)).astype('float')
    # Set everything that is zero to -1
    rademacher_sequence[np.abs(rademacher_sequence) < 1e-10] = -1.0
    A = circulant(rademacher_sequence) # This is an n_features x n_features mat
    selected_rows = np.random.randint(0, n_features, n_measurements)
    A = A[selected_rows, :]
    return A

def _create_prcm_gaussian(n_measurements, n_features):
    """ Method to create a partial random circulant matrix from a gaussian
    sequence.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    Returns
    -----------------
    Partial random circulant matrix as np.array of size (n_measurements, n_features).
    """
    gaussian = np.random.normal(size=(n_features))
    A = circulant(gaussian) # This is an n_features x n_features mat
    selected_rows = np.random.randint(0, n_features, n_measurements)
    A = A[selected_rows, :]
    return A

def _create_prtm_rademacher(n_measurements, n_features):
    """ Method to create a partial random Toeplitz matrix from a rademacher
    sequence.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    Returns
    -----------------
    Partial random Toeplitz matrix as np.array of size (n_measurements, n_features).
    """
    rademacher_sequence = np.random.binomial(1, 0.5,
                                        size=(n_features)).astype('float')
    # Set everything that is zero to -1
    rademacher_sequence[np.abs(rademacher_sequence) < 1e-10] = -1.0
    A = toeplitz(rademacher_sequence) # This is an n_features x n_features mat
    selected_rows = np.random.randint(0, n_features, n_measurements)
    A = A[selected_rows, :]
    return A

def _create_prtm_gaussian(n_measurements, n_features):
    """ Method to create a partial random Toeplitz matrix from a gaussian
    sequence.

    Parameters
    ---------------
    n_measurements : python Integer
        Number of measurements

    n_features : python Integer
        Number of features

    random_seed : pthon Integer
        Random seed for numpy

    random_state : np.random_state
        Random state for numpy

    scaling: Determines scaling of the sampling matrix. Two possibilities:
        a) 'measurements': Matrix is rescaled by 1/sqrt(n_measurements),
                           making correlation independent of n_measurements.
        b) 'normalized_cols': Columns are normalized to have norm 1.

    Returns
    -----------------
    Partial random Toeplitz matrix as np.array of size (n_measurements, n_features).
    """
    gaussian = np.random.normal(size=(n_features))
    A = toeplitz(gaussian) # This is an n_features x n_features mat
    selected_rows = np.random.randint(0, n_features, n_measurements)
    A = A[selected_rows, :]
    return A
