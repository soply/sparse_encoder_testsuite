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
