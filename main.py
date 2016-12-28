# coding: utf8
import getopt
import os
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from calculation_utilities import (calc_relative_error, mutual_incoherence,
                                   symmetric_support_difference)
from encoders.cosamp import cosamp
from encoders.iht import iterative_hard_thresholding
from encoders.lar import least_angle_regression
from encoders.lasso import lasso
from encoders.mp import matching_pursuit
from encoders.omp import orthogonal_matching_pursuit
from encoders.romp import regularized_orthogonal_matching_pursuit
from encoders.sp import subspace_pursuit
from problem_factory import create_specific_problem_data


def recover_support(A, y, u_real, v_real, method, sparsity_level, verbose=True):
    """ Recover the support of u_real given the sampling operator A and the
    measurements y. Method specifies one of the methods

    -lars: Least angle regression
    -lasso: Lasso-path algorithm
    -omp: Orthogonal matching pursuit algorithm

    which is then filtered for the best support of the given sparsity level. For
    lars and omp, there is only 1 support to each sparsity level that is a
    candidate to approximate the support of u_real, but for lars there can be
    more.

    Implementations used from scikit-learn. """
    target_support = np.where(u_real)[0]
    if method == "lar":
        result = least_angle_regression(A, y, sparsity_level)
    elif method == "lasso":
        result = lasso(A, y, target_support, sparsity_level)
    elif method == "mp":
        result = matching_pursuit(A, y, sparsity_level)
    elif method == "omp":
        result = orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "iht":
        result = iterative_hard_thresholding(A, y, sparsity_level,
                                             verbose=True)
    elif method == "romp":
        result = regularized_orthogonal_matching_pursuit(A, y, sparsity_level)
    elif method == "cosamp":
        result = cosamp(A, y, sparsity_level)
    elif method == "sp":
        result = subspace_pursuit(A, y, sparsity_level)
    elif method in ["pomp", "plar", "plasso", "plar", "piht", "promp", "pcosamp",
                    "psp"]:
        # Calculate preconditioned system
        start_time = timer()
        U, S, V = np.linalg.svd(A)
        y_new = np.diag(1.0 / S).dot(U.T).dot(y)
        Pi = np.zeros(A.shape)
        np.fill_diagonal(Pi, 1.0)
        V_lp = Pi.dot(V)
        # Call method again without p in method name
        result = recover_support(V_lp, y_new, u_real, v_real, method[1:],
                                 sparsity_level, verbose=verbose)
        elapsed_time = start_time - timer()  #  Replace time with total time
        # auxV = V.T[:,0:A.shape[0]]
        # auxvec = np.diag(1.0/S).dot(U.T).dot(y)
        # import pdb
        # pdb.set_trace()
        # print "Expected order: ", np.argsort(np.abs(auxV.dot(auxvec)))
        return result[0], result[1], result[2], elapsed_time, result[4]
    else:
        raise RuntimeError('''Specified method not found. Use 'mp', 'omp', 'lar',
            'iht', 'lasso', 'cosamp', 'sp'..''')
    coefs, elapsed_time, support = result
    relative_error = calc_relative_error(coefs, u_real)
    success = np.array_equal(support, target_support)
    if verbose:
        print "Finished example with method {0} in {1} seconds.".format(method,
                                                                        elapsed_time)
        print "Recovered: {0}   Correct: {1}".format(support, target_support)
        print "Success: {0}".format(success)
        print "Error: {0}".format(relative_error)
    return success, support, target_support, elapsed_time, relative_error

# def perform_test_single(filename, M, N, L, c, d, method, noise_type_signal,
#                         largest_signal=None, noise_lev_measurements=0,
#                         noise_type_measurements="gaussian", random_seed = None,
#                         random_state = None, verbosity = None):
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     if random_state is not None:
#         np.random.set_state(random_state)
#     else:
#         random_state = np.random.get_state()
#         A, y, u, v = create_specific_problem_data(M, N, L, c,
#                                                   largest_signal=largest_signal,
#                                                   noise_lev_signal=d,
#                                                   noise_type_signal=noise_type_signal,
#                                                   noise_lev_measurements=noise_lev_measurements,
#                                                   noise_type_measurements=noise_type_measurements,
#                                                   random_seed=random_seed,
#                                                   random_state=random_state)
#         if verbosity:
#             print '''Non zero entries: {0}\n
#                      Entry mean: {1}    Entry absoulte mean: {2}\n
#                      Variance: {2}\n'''.format(u[np.abs(u) > 0], np.mean(u),
#                      np.mean(np.abs(u)), np.var(u))
#         if not os.path.exists(os.path.dirname(filename)):
#             success, support, target_support, elapsed_time, relative_error = \
#                 recover_support(A, y, u, v, method, L)
#             np.savez(filename, support = support, target_support = target_support,
#                      success = success, elapsed_time = elapsed_time,
#                      relative_error = relative_error)
#         else:
#             stored_results = np.load(filename)
#             support = stored_results["support"]
#             target_support = stored_results["target_support"]
#             success = stored_results["success"]
#             elapsed_time = stored_results["elapsed_time"]
#             relative_error = stored_results["relative_error"]
#         return support, target_support, success, elapsed_time, relative_error
#
# def perform_test_batch(basedir_save, M, N, L, c, d, num_tests, method,
#                         noise_type_signal, largest_signal=None,
#                         noise_lev_measurements=0, noise_type_measurements="gaussian",
#                         random_seed = None, random_state = None, verbosity = None):
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     if random_state is not None:
#         np.random.set_state(random_state)
#     else:
#         random_state = np.random.get_state()
#     with open(basedir_save + '/log.txt', "a+") as f:
#         f.write(("Testrun: " + identifier + "\n M: {0} N: {1} L: {2} c: {3}" +
#                  "d: {4} num_tests: {5} random_seed = {6}\n\n").format(M, N, L, c, d,
#                                                                        num_tests, random_seed))
#     batch_results = np.zeros((4, num_tests))
#     for i in range(num_tests):
#         filename = basedir_save + '/' + str(i)
#         support, target_support, success, elapsed_time, relative_error = \
#                 perform_test_single(filename, M, N, L, c, d, method,
#                                     noise_type_signal, largest_signal,
#                                     noise_lev_measurements,
#                                     noise_type_measurements, random_seed,
#                                     random_state, verbosity)
#
#         # Evaluate batch results
#         if success:
#             batch_results[0, i] = 1
#         else:
#             batch_results[0, i] = 0
#         batch_results[1, i] = elapsed_time
#         batch_results[2, i] = relative_error
#         symmetric_diff = symmetric_support_difference(support, target_support)
#         batch_results[3, i] = symmetric_diff
#         np.save(basedir_save + '/meta.npy', batch_results)
#     return batch_results


def perform_test(basedir_save, M, N, L, c, d, num_tests, random_seed, method,
                 noise_type_signal, largest_signal=None, noise_lev_measurements=0,
                 noise_type_measurements="gaussian"):
    if not os.path.exists(basedir_save):
        raise RuntimeError("Directory {0} has not been created yet.".format(
            os.path.dirname(os.path.realpath(__file__)) + basedir_save))

    with open('results/' + identifier + '/log.txt', "a+") as f:
        f.write(("Testrun: " + identifier + "\n M: {0} N: {1} L: {2} c: {3}" +
                 "d: {4} num_tests: {5} random_seed = {6}\n\n").format(M, N, L, c, d,
                                                                       num_tests, random_seed))
    """ meta_results storing the results if support was exactly recovered at the
    end, if the conditions (1) and (2) have been satisfied and if the the
    boundaries for alpha are satisfied. """
    meta_results = np.zeros((4, num_tests))
    np.random.seed(random_seed)
    variance_in_y = np.zeros(num_tests)
    for i in range(num_tests):
        print "\nRun example {0}/{1}".format(i, num_tests)
        random_state = np.random.get_state()
        A, y, u, v = create_specific_problem_data(M, N, L, c,
                                                  largest_signal=largest_signal,
                                                  noise_lev_signal=d,
                                                  noise_type_signal=noise_type_signal,
                                                  noise_lev_measurements=noise_lev_measurements,
                                                  noise_type_measurements=noise_type_measurements,
                                                  random_seed=random_seed,
                                                  random_state=random_state)

        print "Non-zero entries: ", u[np.abs(u) > 0]
        print "Mean: {0}    Abs. mean: {1}".format(np.mean(u), np.mean(np.abs(u)))
        print "Variance: ", np.var(u)
        variance_in_y[i] = np.var(y - A.dot(u))
        print "Variance y - Ax", np.var(y - A.dot(u))
        if not os.path.exists('results/' + identifier + '/' + str(i) +
                              "_support.npy"):
            success, support, target_support, elapsed_time, relative_error = \
                recover_support(A, y, u, v, method, L)
            np.save("results/" + identifier + '/' + str(i) + "_support.npy",
                    np.vstack((support, target_support)))
            np.save("results/" + identifier + '/' + str(i) + "_meta.npy",
                    np.array([success, elapsed_time, relative_error]))
        else:
            supports = np.load("results/" + identifier + '/' + str(i) +
                               "_support.npy")
            support = supports[0, :]
            target_support = supports[1, :]
            meta_info = np.load("results/" + identifier + '/' + str(i) +
                                "_meta.npy")
            success = meta_info[0]
            elapsed_time = meta_info[1]
            relative_error = meta_info[2]

        # Check if support was recovered exactly at the end
        if success:
            meta_results[0, i] = 1
        else:
            meta_results[0, i] = 0

        # Store elapsed time
        meta_results[1, i] = elapsed_time
        # Store error
        meta_results[2, i] = relative_error
        # Symmetric difference
        symmetric_diff = symmetric_support_difference(support,
                                                      target_support)
        meta_results[3, i] = symmetric_diff
        # Store meta results
        np.save("results/" + identifier + "/" + "meta.npy",
                meta_results)
    return meta_results


def perform_test(M, N, L, c, d, num_tests, identifier, random_seed, method,
                 noise_type_signal, largest_signal=None, noise_lev_measurements=0,
                 noise_type_measurements="gaussian"):
    identifier = identifier + '_' + method
    if not os.path.exists('results/' + identifier + '/'):
        os.makedirs('results/' + identifier + '/')
    with open('results/' + identifier + '/log.txt', "a+") as f:
        f.write(("Testrun: " + identifier + "\n M: {0} N: {1} L: {2} c: {3}" +
                 "d: {4} num_tests: {5} random_seed = {6}\n\n").format(M, N, L, c, d,
                                                                       num_tests, random_seed))
    """ meta_results storing the results if support was exactly recovered at the
    end, if the conditions (1) and (2) have been satisfied and if the the
    boundaries for alpha are satisfied. """
    meta_results = np.zeros((4, num_tests))
    np.random.seed(random_seed)
    variance_in_y = np.zeros(num_tests)
    for i in range(num_tests):
        print "\nRun example {0}/{1}".format(i, num_tests)
        random_state = np.random.get_state()
        A, y, u, v = create_specific_problem_data(M, N, L, c,
                                                  largest_signal=largest_signal,
                                                  noise_lev_signal=d,
                                                  noise_type_signal=noise_type_signal,
                                                  noise_lev_measurements=noise_lev_measurements,
                                                  noise_type_measurements=noise_type_measurements,
                                                  random_seed=random_seed,
                                                  random_state=random_state)

        print "Non-zero entries: ", u[np.abs(u) > 0]
        print "Mean: {0}    Abs. mean: {1}".format(np.mean(u), np.mean(np.abs(u)))
        print "Variance: ", np.var(u)
        variance_in_y[i] = np.var(y - A.dot(u))
        print "Variance y - Ax", np.var(y - A.dot(u))
        if not os.path.exists('results/' + identifier + '/' + str(i) +
                              "_support.npy"):
            success, support, target_support, elapsed_time, relative_error = \
                recover_support(A, y, u, v, method, L)
            np.save("results/" + identifier + '/' + str(i) + "_support.npy",
                    np.vstack((support, target_support)))
            np.save("results/" + identifier + '/' + str(i) + "_meta.npy",
                    np.array([success, elapsed_time, relative_error]))
        else:
            supports = np.load("results/" + identifier + '/' + str(i) +
                               "_support.npy")
            support = supports[0, :]
            target_support = supports[1, :]
            meta_info = np.load("results/" + identifier + '/' + str(i) +
                                "_meta.npy")
            success = meta_info[0]
            elapsed_time = meta_info[1]
            relative_error = meta_info[2]

        # Check if support was recovered exactly at the end
        if success:
            meta_results[0, i] = 1
        else:
            meta_results[0, i] = 0

        # Store elapsed time
        meta_results[1, i] = elapsed_time
        # Store error
        meta_results[2, i] = relative_error
        # Symmetric difference
        symmetric_diff = symmetric_support_difference(support,
                                                      target_support)
        meta_results[3, i] = symmetric_diff
        # Store meta results
        np.save("results/" + identifier + "/" + "meta.npy",
                meta_results)
    return meta_results


def show_meta_results(identifier):
    meta_results = np.load("results/" + identifier + "/" + "meta.npy")
    num_tests = meta_results.shape[1]
    # Calculate percentages if necessary
    success_percentage = np.sum(meta_results[0, :]) / float(len(meta_results[0, :]))
    # Calculate average symmetric difference
    average_symmetric_difference = np.mean(meta_results[3, meta_results[0, :] == 0])
    # Present percentages
    print "================== META RESULTS ======================"
    print "1) Percentages:"
    print "Support at the end recovered: {0}".format(success_percentage)
    print "\n2) Timings:"
    print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
        np.mean(meta_results[1, :]),
        np.var(meta_results[1, :]),
        [np.min(np.percentile(meta_results[1, :], 0.95)),
         np.max(np.percentile(meta_results[1, :], 95))],
        np.min(meta_results[1, :]),
        np.max(meta_results[1, :]))
    print "\n3) Relative error:"
    print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
        np.mean(meta_results[2, :]),
        np.var(meta_results[2, :]),
        [np.min(np.percentile(meta_results[2, :], 0.95)),
         np.max(np.percentile(meta_results[2, :], 95))],
        np.min(meta_results[2, :]),
        np.max(meta_results[2, :]))
    if len(meta_results[2, meta_results[0, :] == 1]) > 0:
        print "For successful cases only:"
        print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
            np.mean(meta_results[2, meta_results[0, :] == 1]),
            np.var(meta_results[2, meta_results[0, :] == 1]),
            [np.min(np.percentile(meta_results[2, meta_results[0, :] == 1], 0.95)),
                np.max(np.percentile(meta_results[2, meta_results[0, :] == 1], 95))],
            np.min(meta_results[2, meta_results[0, :] == 1]),
            np.max(meta_results[2, meta_results[0, :] == 1]))
    # Present single index cases
    print "\n4) Suspicious cases:"
    print "Examples support not correct: {0}".format(np.where(meta_results[0, :] == 0)[0])
    print "Symmetric differences: {0}".format(meta_results[3, meta_results[0, :] == 0])
    print "Average symmetric difference for wrong examples: ", average_symmetric_difference


def plot_success_wrt_sparsity_param_matrixdimensions(s, matdim, identifier, method, snr):
    success_percentages = np.zeros((len(s), len(matdim)))
    success_percentages2 = np.zeros((len(s), len(matdim)))
    success_percentages3 = np.zeros((len(s), len(matdim)))
    success_percentages4 = np.zeros((len(s), len(matdim)))
    plt.figure()
    plt.axis([1, 24, 0, 1.1])
    identifier = "visualize_folding_realwithout"
    identifier2 = "visualize_folding_without"
    identifier3 = "results_trondheim"
    identifier4 = "results_trondheim_unfolded"
    colors = ['r', 'b', 'g', 'c', 'y']
    for i in range(1, len(matdim)):
        for j in range(len(s)):
            meta_results = np.load("results/" + identifier + "_" + str(j) + "_" +
                                   str(i) + "_" + method + "/" + "meta.npy")
            meta_results2 = np.load("results/" + identifier2 + "_" + str(j) + "_" +
                                    str(i) + "_" + method + "/" + "meta.npy")
            try:
                meta_results3 = np.loadtxt("results/" + identifier3 + "_" + str(j) + "_" +
                                           str(i) + "/" + "meta.gz")
            except IOError:
                meta_results3 = np.ones((9, 100))
            try:
                meta_results4 = np.loadtxt("results/" + identifier4 + "_" + str(j) + "_" +
                                           str(i) + "/" + "meta.gz")
            except IOError:
                meta_results4 = np.ones((9, 100))
            success_percentages3[j, i] = len(np.where(meta_results3[5, :] == 0)[
                                             0]) / float(len(meta_results3[5, :]))
            success_percentages[j, i] = np.sum(meta_results[0, :]) / float(len(meta_results[0, :]))
            success_percentages2[j, i] = np.sum(
                meta_results2[0, :]) / float(len(meta_results2[0, :]))
            success_percentages4[j, i] = np.sum(
                meta_results4[5, :] == 0) / float(len(meta_results4[5, :] == 0))
        plt.plot(s, success_percentages[:, i], (colors[i] + "-o"), linewidth=4,
                 label="{0} x {1}".format(matdim[i][0], matdim[i][1]))
        plt.plot(s, success_percentages2[:, i], (colors[i] + "--"),  linewidth=4,
                 label="{0} x {1} NF".format(matdim[i][0], matdim[i][1]))
        plt.plot(s, success_percentages4[:, i], (colors[i] + ":"), linewidth=4,
                 label="{0} x {1} MP".format(matdim[i][0], matdim[i][1]))
        plt.plot(s, success_percentages3[:, i], (colors[i] + "-."),  linewidth=4,
                 label="{0} x {1} MP NF".format(matdim[i][0], matdim[i][1]))

    plt.xlabel('Sparsity')
    plt.ylabel('Successful recovery [%]')
    plt.legend(loc="lower left")
    plt.title("Effect of noise folding with original SNR = {1}".format(method, snr))
    plt.show()


def main(argv):
    task = ''
    identifier = ''
    method = ''
    try:
        opts, args = getopt.getopt(argv, "ht:i:p:m:", ["task=", "identifier=", "method="])
    except getopt.GetoptError:
        print "================================================================="
        print 'main.py -t <task> -i <identifier> -m <method>'
        print "Help docstring"
        print "================================================================="
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print "==========================================================="
            print 'Help docstring'
            print "==========================================================="
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-i", "--identifier"):
            identifier = arg
        elif opt in ("-m", "--method"):
            method = arg
    print 'Task: ', task
    print 'Identifier: ', identifier
    print 'Method: ', method
    if task == 'show_results':
        identifier = identifier + '_' + method
        show_meta_results(identifier)
    elif task == "plot_N":
        N = np.array([1200, 2400, 4800, 9600, 9600, 15000])
        M = np.array([500, 750, 1000])
        plot_N_dependent_curve(identifier, N, M, method)
    elif task == 'run_batch':
        N = [(256, 1028), (256, 256), (256, 256),  (256, 256)]
        sparsity_levels = [6]
        c_real = 1.5
        c_max = 4.5
        noise_lev_signal = 0.0
        noise_lev_measurements = 0.2
        random_seeds = [123, 234, 345, 456]
        # N = [(40,200), (80, 200), (180,200), (400, 2250)]
        # # N = [(200, 2250)]
        # sparsity_levels = np.arange(1,25)
        # c_real = 1.5
        # c_max = 4.5
        # d_real = 0.25
        # random_seeds = sparsity_levels
        noise_type_signal = "linf_bounded"
        # for i in range(len(sparsity_levels)):
        for j in range(len(N)):
            perform_test(N[j][0], N[j][1], sparsity_levels[j], c_real,
                         noise_lev_signal, 100, identifier + '_' + str(j) + '_' + str(j),
                         random_seeds[j], method,
                         noise_type_signal=noise_type_signal,
                         largest_signal=c_max,
                         noise_lev_measurements=noise_lev_measurements,
                         noise_type_measurements="gaussian")
        # plot_success_wrt_sparsity_param_matrixdimensions(sparsity_levels, N, identifier,
        #     method, c_real/d_real)
    else:
        return

if __name__ == "__main__":
    main(sys.argv[1:])
