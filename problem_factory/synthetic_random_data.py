# coding: utf8
""" Methods to create synthetic random data of type A(u+v) = y + (epsilon) """

__author__ = "Timo Klock"

import numpy as np


def create_specific_problem_data_from_problem(problem):
    """ Method to create specific problem from a problem dictionary. Essentially
    unpacks the problem dictionary and calls 'create_specific_problem_data'.
    Therefore, we only explain the dictionary keys here. To understand how the
    data is created from this, check the methods 'create_specific_problem_data',
    'create_sampling_matrix', 'create_sparse_signal', 'create_noise'.

    Methodology
    -------------
    A : Created by 'create_sampling_matrix'.
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
    A = problem.get("A", None)
    random_seed = problem.get("random_seed", None)
    random_state = problem.get("random_state", None)
    return create_specific_problem_data(n_measurements, n_features,
                                        sparsity_level,
                                        smallest_signal, largest_signal,
                                        noise_type_signal,
                                        noise_lev_signal,
                                        noise_type_measurements,
                                        noise_lev_measurements,
                                        A, random_seed, random_state)


def create_specific_problem_data(n_measurements, n_features, sparsity_level,
                                 smallest_signal, largest_signal=None,
                                 noise_type_signal="uniform_ensured_max",
                                 noise_lev_signal=0,
                                 noise_type_measurements="gaussian",
                                 noise_lev_measurements=0,
                                 A=None, random_seed=None, random_state=None):
    """ Method to create specific problem data A, y+epsilon, u, v with
                A(u+v) = y + epsilon.
    Uses the methods below to create the data with the respective properties.
    For documentation on the (most of the) inputs, check methods
    'create_sampling_matrix', 'create_sparse_signal', 'create_noise'.

    Methodology
    -------------
    A : Created by 'create_sampling_matrix'.
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
        A = create_sampling_matrix(n_measurements, n_features)
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

def create_sampling_matrix(n_measurements, n_features, random_seed=None,
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
    np.array of size (n_measurements, n_features).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    A = np.random.randn(n_measurements, n_features)
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

def create_sparse_signal(n_features, sparsity_level, smallest_signal,
                         largest_signal=None, random_seed=None,
                         random_state=None):
    """ Method to create a sparse signal of size (n_features, 1) with given
    support size. Entries are created either via standard normal distribution
    that is rescaled such that the smallest entry is equal to 'smallest_signal'
    or, if 'largest_signal' is given, via uniform sampling in 'smallest_signal'
    to 'largest_signal' and randomly assigning signs to the entries.

    Parameters
    ---------------
    n_features : python Integer
        Number of features

    sparsity_level : python Integer
        Support size of the signal

    smallest_signal : python float
        Smallest signal entry will always be equal to this number.

    largest_signal : python float
        Optional, if given, largest signal entry will be equal to this number.

    random_seed : pthon Integer
        Random seed for numpy

    random_state : np.random_state
        Random state for numpy

    Returns
    -----------------
    np.array of shape (n_features, 1) of floats.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    signal = np.zeros(n_features)
    indices = np.random.permutation(list(range(0, n_features - 1)))
    if largest_signal is None:
        aux = np.random.randn(sparsity_level, 1)
        scaling_factor = smallest_signal / np.min(np.abs(aux))
        signal[indices[0:sparsity_level]] = scaling_factor * aux
    else:
        entries = np.random.uniform(low=smallest_signal,
                                    high=largest_signal,
                                    size=(sparsity_level,))
        # Ensure minimum entry is equal to c
        entries[np.random.choice(sparsity_level)] = smallest_signal
        sign = np.random.choice([1.0, -1.0], size=(sparsity_level,))
        signal[indices[0:sparsity_level]] = np.multiply(entries, sign)
    return signal

def create_noise(length, noise_level, noise_type, random_seed=None,
                 random_state=None):
    """ Method to create a noise vector of size 'length' with a predefined noise
    level and predefined noise_type.

    Parameters
    ---------------
    length : python Integer
        Length of the noise vector

    noise_level : python Float
        Strength of the noise. Concrete meaning depends on the noise type (see
        noise_type explanation for more information).

    noise_type : python String
        Defines the type of noise. Following options are possible:
        a) "uniform": Uniform noise between (-noise_level, +noise_level)
        b) "uniform_ensured_max": Uniform noise between (-noise_level,
                                       +noise_level) but globally rescaled such
                                       that the maximum entry is equal to
                                       noise_level.
        c) "gaussian": Gaussian noise with variance given by noise level.
        d) "gaussian_ensured_max": Standard normal gaussian noise rescaled such
                                   that the maximum entry equals noise_level.

    random_seed : pthon Integer
        Random seed for numpy

    random_state : np.random_state
        Random state for numpy

    Returns
    -----------------
    np.array of shape (length, 1) of floats.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    if noise_type == "uniform":
        noise = np.random.uniform(low=-noise_level, high=noise_level,
                                      size=length)
    elif noise_type == "uniform_ensured_max":
        aux_noise = np.random.uniform(low=-noise_level, high=noise_level,
                                      size=length)
        scaling_factor_noise = noise_level / np.max(np.abs(aux_noise))
        noise = aux_noise * scaling_factor_noise
    elif noise_type == "gaussian":
        noise = np.random.normal(scale=noise_level ** 2, size=length)
    elif noise_type == "gaussian_ensured_max":
        aux_noise = np.random.normal(size=length)
        scaling_factor_noise = noise_level / np.max(np.abs(aux_noise))
        noise = aux_noise * scaling_factor_noise

    return noise
