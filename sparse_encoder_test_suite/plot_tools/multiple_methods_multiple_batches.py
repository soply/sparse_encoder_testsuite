# coding: utf8
""" Plotting methods for experiments with synthethic random data where we test
    multiple batches under multiple methods. """

__author__ = "Timo Klock"
import json
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__marker_rotation__ = ['o', 'H', 's', '^', 'None', '+', '.', 'D', 'x']
__linestyle_rotation__ = ['-', '--', ':', '-.']

def success_vs_sparsity_level(basefolder, identifier, methods,
                              alternative_keys = None, title = None,
                              xlabel = None, ylabel = None, save_as = None,
                              leg_loc = 'lower right', legend_entries = None):
    """ Creates a plot success rate vs sparsity level plot. Sparsity level is
    on x axis and success rate on y. The method akes a list of methods as an
    input, so several methods can be compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_sparsity_level>/meta.npz"

    where <ctr_sparsity_level> is a counter from 0 to the number of different
    sparsity levels that can be found.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['lar', 'omp', 'lasso'].

    alternative_keys, optional : python list of strings
        If given, must be of length of 'methods' and specifies the key of the
        meta results that is used as the 'success' indicator. Per default this
        is either the key "success" or "tiling_contains_real" (depending on
        which method is used), but if stopping criteria shall be tested, other
        keys may be desired.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.

    legend_entries, optional : List of strings
        List of strings of the same size as methods (if given), yielding legend
        entries.
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    sparsity_levels = problem['sparsity_level']
    success_rates = np.zeros((len(sparsity_levels), len(methods)))
    for i, method in enumerate(methods):
        for j, sparsity_level in enumerate(sparsity_levels):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            if alternative_keys is not None:
                success_rates[j, i] = np.sum(meta_results[alternative_keys[i]])/ \
                                             float(num_tests)
            elif "success" in meta_results.keys():
                success_rates[j, i] = np.sum(meta_results["success"])/ \
                                                    float(num_tests)
            elif "tiling_contains_real" in meta_results.keys():
                # Key for our multi-penalty framework
                success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                    float(num_tests)
            else:
                raise RuntimeError("Can not find key for success rate in" + \
                    " results. Keys are: {0}".format(meta_results.keys()))
    fig = plt.figure(figsize = (16,9))
    for j in range(success_rates.shape[1]):
        plt.plot(sparsity_levels, success_rates[:,j], linewidth = 3.0,
            linestyle = __linestyle_rotation__[j % len(__linestyle_rotation__)],
                marker = __marker_rotation__[j % len(__marker_rotation__)],
                markersize = 15.0)
    if legend_entries is None:
        legend_entries = [r''+ method.replace('_','').upper() for method in methods]
    plt.legend(legend_entries, loc = leg_loc, ncol = 2, fontsize = 20.0)
    if xlabel is None:
        plt.xlabel(r'Support size')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs support size')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_vs_signal_noise(basefolder, identifier, methods,
                            alternative_keys = None, title = None,
                            xlabel = None, ylabel = None, save_as = None,
                            leg_loc = 'lower right', legend_entries = None):
    """ Creates a plot success rate vs signal to noise ration (SNR) where the
    noise is applied directly to the signal (not on the measurements). SNR is
    on x axis and success rate on y. The method akes a list of methods as an
    input, so several methods can be compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_SNR>/meta.npz"

    where <ctr_SNR> is a counter from 0 to the number of different signal to
    noise ratios.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['lar', 'omp', 'lasso'].

    alternative_keys, optional : python list of strings
        If given, must be of length of 'methods' and specifies the key of the
        meta results that is used as the 'success' indicator. Per default this
        is either the key "success" or "tiling_contains_real" (depending on
        which method is used), but if stopping criteria shall be tested, other
        keys may be desired.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.

    legend_entries, optional : List of strings
        List of strings of the same size as methods (if given), yielding legend
        entries.
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    signal_noise = np.array(problem['noise_lev_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_to_noise_ratios = smallest_signal_entry/signal_noise
    success_rates = np.zeros((len(signal_to_noise_ratios), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_to_noise_ratios)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            if alternative_keys is not None:
                success_rates[j, i] = np.sum(meta_results[alternative_keys[i]])/ \
                                             float(num_tests)
            elif "success" in meta_results.keys():
                success_rates[j, i] = np.sum(meta_results["success"])/ \
                                                    float(num_tests)
            elif "tiling_contains_real" in meta_results.keys():
                # Key for our multi-penalty framework
                success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                    float(num_tests)
            else:
                raise RuntimeError("Can not find key for success rate in" + \
                    " results. Keys are: {0}".format(meta_results.keys()))
    fig = plt.figure(figsize = (16,9))
    for j in range(success_rates.shape[1]):
        plt.semilogx(signal_to_noise_ratios, success_rates[:,j], linewidth = 3.0,
                linestyle = __linestyle_rotation__[j % len(__linestyle_rotation__)],
                marker = __marker_rotation__[j % len(__marker_rotation__)],
                markersize = 15.0)
    if legend_entries is None:
        legend_entries = [r''+ method.replace('_','').upper() for method in methods]
    plt.legend(legend_entries, loc = leg_loc, ncol = 2, fontsize = 20.0)
    if xlabel is None:
        plt.xlabel(r'SNR (smallest signal entry/signal noise)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs SNR')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_vs_signal_gap(basefolder, identifier, methods, alternative_keys = None,
                          title = None, xlabel = None, ylabel = None,
                          save_as = None, leg_loc = 'lower right',
                          legend_entries = None):
    """ Creates a plot success rate vs signal gap meaning the largest signal
    divided through the smallest signal appearning in the randoms signal. The
    signal gap  is on x axis and success rate on y.
    The method akes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_signal_gap>/meta.npz"

    where <ctr_signal_gap> is a counter from 0 to the number of different signal
    to noise ratios.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['lar', 'omp', 'lasso'].

    alternative_keys, optional : python list of strings
        If given, must be of length of 'methods' and specifies the key of the
        meta results that is used as the 'success' indicator. Per default this
        is either the key "success" or "tiling_contains_real" (depending on
        which method is used), but if stopping criteria shall be tested, other
        keys may be desired.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.

    legend_entries, optional : List of strings
        List of strings of the same size as methods (if given), yielding legend
        entries.
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    largest_signal_entry = np.array(problem['largest_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_gaps = largest_signal_entry/smallest_signal_entry
    success_rates = np.zeros((len(signal_gaps), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_gaps)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            if alternative_keys is not None:
                success_rates[j, i] = np.sum(meta_results[alternative_keys[i]])/ \
                                             float(num_tests)
            elif "success" in meta_results.keys():
                success_rates[j, i] = np.sum(meta_results["success"])/ \
                                                    float(num_tests)
            elif "tiling_contains_real" in meta_results.keys():
                # Key for our multi-penalty framework
                success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                    float(num_tests)
            else:
                raise RuntimeError("Can not find key for success rate in" + \
                    " results. Keys are: {0}".format(meta_results.keys()))
    fig = plt.figure(figsize = (16,9))
    for j in range(success_rates.shape[1]):
        plt.semilogx(signal_gaps, success_rates[:,j], linewidth = 3.0,
                linestyle = __linestyle_rotation__[j % len(__linestyle_rotation__)],
                marker = __marker_rotation__[j % len(__marker_rotation__)],
                markersize = 15.0)
    if legend_entries is None:
        legend_entries = [r''+ method.replace('_','').upper() for method in methods]
    plt.legend(legend_entries, loc = leg_loc, ncol = 2, fontsize = 20.0)
    if xlabel is None:
        plt.xlabel(r'Signal gap (Largest absoulute entry/Smallest absolute entry)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs Signal gap')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()


def success_vs_measurement_noise(basefolder, identifier, methods,
                                 alternative_keys = None,
                                 title = None, xlabel = None, ylabel = None,
                                 save_as = None, leg_loc = 'lower right',
                                 legend_entries = None):
    """ Creates a plot success rate vs measurement noise. The
    measurement nose is logarithmically on x axis and success rate on y.
    The method akes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_measurement_noise>/meta.npz"

    where <ctr_measurement_noise> is a counter from 0 to the number of different
    measurement noises.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['lar', 'omp', 'lasso'].

    alternative_keys, optional : python list of strings
        If given, must be of length of 'methods' and specifies the key of the
        meta results that is used as the 'success' indicator. Per default this
        is either the key "success" or "tiling_contains_real" (depending on
        which method is used), but if stopping criteria shall be tested, other
        keys may be desired.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.

    legend_entries, optional : List of strings
        List of strings of the same size as methods (if given), yielding legend
        entries.
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    smallest_signal = problem['smallest_signal']
    measurement_noises = np.array(problem['noise_lev_measurements'])
    measurement_SNR = measurement_noises/smallest_signal
    success_rates = np.zeros((len(measurement_SNR), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(measurement_SNR)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            if alternative_keys is not None:
                success_rates[j, i] = np.sum(meta_results[alternative_keys[i]])/ \
                                             float(num_tests)
            elif "success" in meta_results.keys():
                success_rates[j, i] = np.sum(meta_results["success"])/ \
                                                    float(num_tests)
            elif "tiling_contains_real" in meta_results.keys():
                # Key for our multi-penalty framework
                success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                    float(num_tests)
            else:
                raise RuntimeError("Can not find key for success rate in" + \
                    " results. Keys are: {0}".format(meta_results.keys()))
    fig = plt.figure(figsize = (16,9))
    for j in range(success_rates.shape[1]):
        plt.semilogx(measurement_SNR, success_rates[:,j], linewidth = 3.0,
                linestyle = __linestyle_rotation__[j % len(__linestyle_rotation__)],
                marker = __marker_rotation__[j % len(__marker_rotation__)],
                markersize = 15.0)
    if legend_entries is None:
        legend_entries = [r''+ method.replace('_','').upper() for method in methods]
    plt.legend(legend_entries, loc = leg_loc, ncol = 2, fontsize = 20.0)
    if xlabel is None:
        plt.xlabel(r'$\sigma^2/c_{\rm{min}}$')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs measurement noise level $\sigma^2$')
    else:
        plt.title(title)
    plt.xlim([np.min(measurement_noises)-1e-5, np.max(measurement_noises)+1e-5])
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()
