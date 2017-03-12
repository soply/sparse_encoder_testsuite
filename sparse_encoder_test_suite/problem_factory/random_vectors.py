# coding: utf8
""" Methods to create random vectors, either to simulate a signal (ie. a sparse
    vector), or to create a signal/measurement noise vector. Different ways of
    creating noise vectors are available (see docs for create_noise below). """

__author__ = "Timo Klock"

import numpy as np
from random_matrices import create_sampling_matrix

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
    else:
        raise NotImplementedError("Unknown noise type {0}. Please choose" + \
            " from the list 'uniform', 'uniform_ensured_max', 'gaussian'" + \
            ", 'gaussian_ensured_max'.")
    return noise
