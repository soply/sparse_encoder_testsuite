#coding: utf8
import numpy as np

def relative_error(u1, u2):
    """ Doc """
    if u1.shape != u2.shape:
        raise RuntimeError("relative_error shape mismatch:{0} and {1}".format(
            u1.shape, u2.shape))
    return np.linalg.norm(u1 - u2) / np.linalg.norm(u2)

def symmetric_support_difference(support1, support2):
    """ Adds indices that belong to either the support of u_calc but not the
    support of u_real, or vice versa to the support of u_real but not the
    support of u_calc to a list.

    Returns the list of non-overlapping indices as well as the length of
    non-overlapping nodes. """
    return len(np.setdiff1d(support1, support2)) + \
           len(np.setdiff1d(support2, support1))
