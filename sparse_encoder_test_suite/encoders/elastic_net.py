#coding: utf8
from timeit import default_timer as timer

import numpy as np
from sklearn.linear_model import enet_path
from ..calculation_utilities.general import find_support_minimalSD


def elastic_net(A, y, target_support, sparsity_level, tau_min, tau_max,
                n_tau, tau_scaling, **kwargs):
    """ Performs a path algorithm for elastic net regularization with a grid
    specified by tau_min, tau_max, n_tau, and tau_scaling, that describes a
    grid for weighting the l1 and l2 penalty terms. For all support that are
    retrieved we check if one of the supports equals the target_support. This
    support is returned. If the real support wasn't found by any regularization
    parameter combination, an arbitrary support is returned.
    """
    suppress_warning =  kwargs.get('suppress_warning', False)
    max_iter = kwargs.get('max_iter_enet', 3 * sparsity_level + 10)
    start = timer()
    # Create tau scaling range
    if tau_scaling == 'linscale':
        tau_range = np.linspace(tau_min, tau_max, n_tau)
    elif tau_scaling == 'logscale':
        tau_range = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    else:
        raise NotImplementedError('enet: Choose linscale or logscale as tau_scaling.')
    coefs = np.zeros((A.shape[1], 0))
    for tau in tau_range:
        alphas, new_coefs, dual_gaps = enet_path(A, y, l1_ratio = tau,
                                             n_alphas = max_iter,
                                             eps = 1e-5,
                                             fit_intercept = True)
        coefs = np.concatenate((coefs, new_coefs), axis = 1)
    binary_coefs = coefs.astype('bool')
    support, index = find_support_minimalSD(binary_coefs, target_support)
    coefs = coefs[:, index]
    elapsed_time = timer() - start
    return coefs, elapsed_time, support
