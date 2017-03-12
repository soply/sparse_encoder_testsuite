# coding: utf8
""" Plotting methods for experiments with synthethic random data where we test
    multiple batches under multiple methods. """

__author__ = "Timo Klock"
import json

import matplotlib.pyplot as plt
import numpy as np


def success_vs_sparsity_level(basefolder, identifier, methods, title = None,
                              xlabel = None, ylabel = None, save_as = None,
                              leg_loc = 'lower right'):
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
            if "success" in meta_results.keys():
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
    plt.plot(sparsity_levels, success_rates, linewidth = 3.0)
    plt.legend(methods, loc = leg_loc, ncol = 2)
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

def success_vs_signal_noise(basefolder, identifier, methods, title = None,
                            xlabel = None, ylabel = None, save_as = None,
                            leg_loc = 'lower right'):
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
            if "success" in meta_results.keys():
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
    plt.semilogx(signal_to_noise_ratios, success_rates, linewidth = 3.0)
    plt.legend(methods, loc = leg_loc, ncol = 2)
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
