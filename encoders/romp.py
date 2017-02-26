# coding: utf8
from timeit import default_timer as timer

import numpy as np

from ..calculation_utilities.least_squares_regression import \
                                                        least_squares_regression


def regularized_orthogonal_matching_pursuit(A, y, sparsity_level, **kwargs):
    """ Regularised orthogonal matching pursuit algorithm.
    FIXME: Current implementation does not work. There must be a mistake, but
           could not been found yet.
    """
    # Default choice: discrepancy principle with tau = 1
    tolerance = kwargs.get('stopping_tolerance', 1e-13)
    M, N = A.shape
    start_time = timer()
    support = set()
    residual = y
    while np.linalg.norm(residual) > tolerance:
        # Identify step
        observation = np.abs(A.T.dot(residual))
        if np.count_nonzero(observation) > sparsity_level:
            entry_order = np.argsort(observation).astype("int")
            active_set = entry_order[-(sparsity_level):]
        else:
            active_set = np.where(oberservation)[0]
        # Regularise step
        ordered_observation = observation[active_set][::-1]
        print "active_set", active_set
        active_set = active_set[::-1]
        start_idx = 0
        end_idx = 0
        current_max_energy = 0
        # Initialize with whole range
        start_max = 0
        end_max = len(ordered_observation)
        max_entry = ordered_observation[start_idx]
        while end_idx < len(ordered_observation):
            if ordered_observation[end_idx] < 0.5 * ordered_observation[start_idx]:
                subset_energy = np.linalg.norm(ordered_observation[start_idx:end_idx],
                                               ord=2)
                if subset_energy > current_max_energy:
                    current_max_energy = subset_energy
                    start_max = start_idx
                    end_max = end_idx
                    start_idx = end_idx
                print "ordered_observation ", ordered_observation
            else:
                end_idx = end_idx + 1
        else:
            subset_energy = np.linalg.norm(ordered_observation[start_idx:end_idx],
                                           ord=2)
            if subset_energy > current_max_energy:
                current_max_energy = subset_energy
                start_max = start_idx
                end_max = end_idx
            start_idx = start_idx + 1
        support = support.union(active_set[start_max:end_max])
        least_squares_sol = least_squares_regression(A, y, support=list(support))
        residual = y - A.dot(least_squares_sol)
        print support, np.linalg.norm(residual)

    elapsed_time = timer() - start_time
    solution = np.zeros(A.shape[1])
    solution[list(support)] = least_squares_sol
    return solution, elapsed_time, sorted(list(support))
