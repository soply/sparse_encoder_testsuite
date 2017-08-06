# coding: utf8
""" General utility methods """

__author__ = "Timo Klock"

import numpy as np


def relative_error(u1, u2):
    """ Computes the relative l2 error between u1 and u2. Throws RuntimeError
    if shapes are mismatched.

    Parameters
    ------------
    u1 : np.array, shape(n, m)
        First array.

    u2 : np.array, shape(n, m)
        Second array.

    Returns
    -------------
    Float number with relative l2 norm between u1 and u2.
    """
    if u1.shape != u2.shape:
        raise RuntimeError("relative_error shape mismatch:{0} and {1}".format(
            u1.shape, u2.shape))
    return np.linalg.norm(u1 - u2) / np.linalg.norm(u2)

def find_support_minimalSD(candidates, target_support):
    """
    Method to find the support with the minimal distance to the target support
    in symmetric difference (must not be unique, then an arbitrary support is
    chosen). The matrix candidates contains columnwise supports as a binary
    matrix, ie. candidates[j,i] == 1 means that the i-th candidates has an
    active entry at position j. Target support is given as an array of integers.

    Parameters
    --------------
    candidates : np.array with bools, shape(n, n_candidates)
        Columnwise candidates for supports. Candidates are formatted as binary
        vectors of fixed size n (signal length) where a 1 indicates an
        active entry and a 0 an inactive entry.

    target_support : np.array with Integer, shape(n_support_size)
        Array that stores positions of active entries as integers.

    Returns
    --------------
    Tuple with two entries:
    1st: Support from candidates with minimum symmetric difference (arbitrary, if
    several with same differences exist).
    2nd: Column index of candidates that provides the support with minimum SD.
    """
    sds = np.zeros(candidates.shape[1]).astype('int')
    for i in range(candidates.shape[1]):
        sds[i] = symmetric_support_difference(np.where(candidates[:,i])[0],
                                              target_support)
    return np.where(candidates[:,np.argmin(sds)])[0], np.argmin(sds)



def symmetric_support_difference(support1, support2):
    """ Counts the entries in the symmetric difference of support1 and support2.

    Parameters
    --------------
    support1 : np.array, shape(n, 1)
        First supprt, given as a 1d np.array with ints.

    support2 : np.array, shape(n, 1)
        First supprt, given as a 1d np.array with ints.

    Returns
    --------------
    Integer corresponding to the length of the symmetric difference of both
    supports.
    """
    return len(np.setdiff1d(support1, support2)) + \
           len(np.setdiff1d(support2, support1))
