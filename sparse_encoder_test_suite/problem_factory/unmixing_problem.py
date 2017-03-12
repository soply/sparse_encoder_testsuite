# coding: utf8
""" Methods to create synthetic random data of type A(u+v) = y + (epsilon) """

__author__ = "Timo Klock"

import numpy as np

from random_matrices import create_sampling_matrix
from random_vectors import create_noise, create_sparse_signal


def create_specific_problem_data_from_problem(problem):
    """ Method to create specific problem from a problem dictionary. Essentially
    unpacks the problem dictionary and calls 'create_specific_problem_data'.
    Therefore, we only explain the dictionary keys here. To understand how the
    data is created from this, check the methods 'create_specific_problem_data',
    'create_sampling_matrix' (from random_matrices.py), 'create_sparse_signal',
    'create_noise'.

    Methodology
    -------------
    A : Created by 'create_sampling_matrix' in random_matrices.py.
    u : Created by 'create_sparse_signal'.
    v : Created by 'create_noise' with noise_type_signal, noise_lev_signal.
    epsilon : Created by 'create_noise' with noise_type_measurements,
              noise_lev_measurements.
    y : Created by A.dot(u+v).

    Parameters
    --------------
    problem : python dictionary with the following (key, value) pairs:

        Dictionary-key | Description
        ---------------------------
        n_measurements | Number of measurements.
        n_features | Number of features.
        sparsity_level | Sparsity level of the correct u in A(u+v) = y + eps
        smallest_signal | Minimal signal strength min(|u_i|, i in supp(u))
        largest_signal | Maximal signal strength max(|u_i|, i in supp(u))
        noise_type_signal | Type of noise that is applied to the signal (ie.
                            type of noise of v).
        noise_lev_signal | Noise level of the signal noise.
        noise_type_measurements | Type of noise that is applied to the
                                  measurements y (ie. type of noise of eps).
        noise_lev_measurements | Noise level of the measurement noise.
        random_seed | Random seed for the data creation. If given and fixed, the
                      same random data is created.
        sampling_matrix_type | Type of sampling matrix. See random_matrices.py
                               in this folder to see available matrices.

    Returns
    -------------
    Problem data {A, y, u, v} where y is given as A.dot(u+v) + epsilon, hence
    the noise is already applied. All objects are np.arrays of sizes
    {(n_measurements, n_features), (n_measurements, 1), (n_features, 1),
    (n_features, 1)}.
    """
    n_measurements = problem["n_measurements"]
    n_features = problem["n_features"]
    sparsity_level = problem["sparsity_level"]
    smallest_signal = problem["smallest_signal"]
    largest_signal = problem.get("largest_signal", None)
    noise_type_signal = problem["noise_type_signal"]
    noise_lev_signal = problem["noise_lev_signal"]
    noise_type_measurements = problem["noise_type_measurements"]
    noise_lev_measurements = problem["noise_lev_measurements"]
    mat_type = problem["sampling_matrix_type"]
    A = problem.get("A", None)
    random_seed = problem.get("random_seed", None)
    random_state = problem.get("random_state", None)
    return create_specific_problem_data(n_measurements, n_features,
                                        sparsity_level,
                                        smallest_signal, mat_type,
                                        largest_signal, noise_type_signal,
                                        noise_lev_signal,
                                        noise_type_measurements,
                                        noise_lev_measurements,
                                        A, random_seed, random_state)


def create_specific_problem_data(n_measurements, n_features, sparsity_level,
                                 smallest_signal, mat_type, largest_signal=None,
                                 noise_type_signal="uniform_ensured_max",
                                 noise_lev_signal=0,
                                 noise_type_measurements="gaussian",
                                 noise_lev_measurements=0,
                                 A=None, random_seed=None, random_state=None):
    """ Method to create specific problem data A, y+epsilon, u, v with
                A(u+v) = y + epsilon.
    Uses the methods below and from random_matrices.py to create the data with
    the respective properties.

    Methodology
    -------------
    A : Created by 'create_sampling_matrix' (from random_matrices.py).
    u : Created by 'create_sparse_signal'.
    v : Created by 'create_noise' with noise_type_signal, noise_lev_signal.
    epsilon : Created by 'create_noise' with noise_type_measurements,
              noise_lev_measurements.
    y : Created by A.dot(u+v).

    Parameters
    --------------
    A : numpy array, shape (n_measurements, n_features)
        If A is already determined and shall not be created, we can give this as
        an input to this function. Then the given A is used to create the
        measurements y.

    Returns
    -------------
    Problem data {A, y, u, v} where y is given as A.dot(u+v) + epsilon, hence
    the noise is already applied. All objects are np.arrays of sizes
    {(n_measurements, n_features), (n_measurements, 1), (n_features, 1),
    (n_features, 1)}.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    if A is None:
        A = create_sampling_matrix(n_measurements, n_features, mat_type)
    u_real = create_sparse_signal(A.shape[1], sparsity_level, smallest_signal,
                                  largest_signal=largest_signal)
    y = A.dot(u_real)
    if noise_lev_signal > 0:
        v_real = create_noise(A.shape[1], noise_lev_signal, noise_type_signal,
            random_state = random_state)
        y = y + A.dot(v_real)
    else:
        v_real = np.zeros(A.shape[1])
    if noise_lev_measurements > 0:
        m_noise = create_noise(A.shape[0], noise_lev_measurements,
            noise_type_measurements, random_state = random_state)
        y = y + m_noise
    return A, y, u_real, v_real
