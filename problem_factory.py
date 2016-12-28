# coding: utf8
import numpy as np


def create_specific_problem_data(n_measurements, n_features, sparsity_level,
                                 smallest_signal, largest_signal=None,
                                 noise_type_signal="linf_bounded", noise_lev_signal=0,
                                 noise_type_measurements="gaussian", noise_lev_measurements=0,
                                 A=None, random_seed=None, random_state=None):
    """ Method to create a specific data set/example. """
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
                         largest_signal=None, random_seed=None, random_state=None):
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
                                    high=largest_signal, size=(sparsity_level,))
        # Ensure minimum entry is equal to c
        entries[np.random.choice(sparsity_level)] = smallest_signal
        sign = np.random.choice([1.0, -1.0], size=(sparsity_level,))
        signal[indices[0:sparsity_level]] = np.multiply(entries, sign)
    return signal

def create_noise(length, noise_level, noise_type, random_seed=None,
                 random_state=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if random_state is not None:
        np.random.set_state(random_state)
    if noise_type == "linf_bounded":
        # aux_noise = np.random.normal(size=length)
        # scaling_factor_noise = noise_level / np.max(np.abs(aux_noise))
        # noise = aux_noise * scaling_factor_noise
        aux_noise = np.random.uniform(low = -1.0, high = 1.0,
            size=length)
        scaling_factor_noise = noise_level / np.max(np.abs(aux_noise))
        noise = aux_noise * scaling_factor_noise
    elif noise_type == "l2_bounded":
        # aux_noise = np.random.normal(size=length)
        # scaling_factor_noise = noise_level / np.linalg.norm(aux_noise, ord=2)
        # noise = aux_noise * scaling_factor_noise
        aux_noise = np.random.uniform(low = -1.0, high = 1.0,
            size=length)
        scaling_factor_noise = noise_level / np.linalg.norm(aux_noise, ord = 2)
        noise = aux_noise * scaling_factor_noise
    elif noise_type == "gaussian":
        noise = np.random.normal(scale=noise_level ** 2, size=length)
    return noise
