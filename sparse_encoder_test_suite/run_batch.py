# coding: utf8
""" Methods to run and analyse repitions of experiments with synthetic random
    data. """

__author__ = "Timo Klock"

import json
import os

import numpy as np

from calculation_utilities.general import symmetric_support_difference
from encoders.handler import check_method_validity, recover_support
from problem_factory.unmixing_problem import \
    create_specific_problem_data_from_problem as create_data_unmixing
from problem_factory.pertubation_problem import \
    create_specific_problem_data_from_problem as create_data_pertubation

__available_problem_types__ = ['unmixing', 'pertubation']

def run_numerous_one_constellation(problem, results_prefix = None):
    """ Create numerous simulations of problems of one of the following types:

        1) A * (u + v) = y + eps
        2) (A + E)u = y + eps

    with randomly created data. The run characteristics (ie.
    noise levels, noise types, signal and noise strength and so forth) are
    given in the dictionary called 'problem'. Also, the
    dictionary stores other important characteristics of the run. Concretely,
    the dictonary must contain the following information:

    Dictionary-key | Description
    ---------------------------
    identifier | Subfolder identifier where the results shall be stored. Full
                 path will be "/results_batch/<method>_<identifier>/", or
                 "<results_prefix>/<method>_<identifier>" if results_prefix
                is given.
    num_tests | Number of runs that shall be performed for the given
                characteristics.
    method | Sparse encoder to use.

    For creation of random data (check the respective files to see what options
    for specific keys are available, and what specific options are used for).

    n_measurements | Number of measurements.
    n_features | Number of features.
    sparsity_level | Sparsity level of the correct u in A(u+v) = y + eps
    smallest_signal | Minimal signal strength min(|u_i|, i in supp(u))
    largest_signal | Maximal signal strength max(|u_i|, i in supp(u))
    noise_type_measurements | Type of noise that is applied to the measurements
                              y (ie. type of noise of eps).
    noise_lev_measurements | Noise level of the measurement noise.
    random_seed | Random seed for the data creation. If given and fixed, the
                  same random data is created.
    verbosity | If false, output will be very minimized.
    sampling_matrix_type | Type of sampling matrix. See random_matrices.py in
                           problem_factory folder to see available matrices.
    problem_type | The type of problem to solve. Problems of type 1) are called
                   'unmixing', problems of type 2) are called 'pertubation'.

    Moreover, dependent on the problem type, the following properties need to be
    specified as well.

    For problems of type 1):
    noise_type_signal | Type of noise that is applied to the signal (ie. type
                        of noise of v).
    noise_lev_signal | Noise level of the signal noise.

    For problems of type 2):
    pertubation_matrix_type | Type of pertubation matrix that is added to
                              A. Can take same values as the sampling matrix.
    pertubation_matrix_level | Scaling factor between pertubation matrix
                               and sampling matrix, ie. ||E||_2/||A||_2.
    solver_parameter (optional) | Python dictionary that may contain additional
                                  parameters that a necessary for a specific
                                  solver. To get more information on solver
                                  parameters, check out the encoders code and
                                  docs of solvers.

    Method will save the results of each single run to a file called i_data.npz
    in the folder 'results_batch/<method>_<identifier>/', or if a 'results_prefix'
    is given, it will be stored in '<results_prefix>/<method>_<identifier>/'.
    If the file already exists, the specific run will be skipped (this is
    useful if we want to stop a run in the middle and restart it). At the end
    of the run, meta results over all runs are created and stored to a file
    called meta.txt in the same folder. This can be used to analyse a specific
    batch of runs.
    """
    if results_prefix is not None:
        resultdir = results_prefix + problem["method"] + "_" + \
                                     problem['identifier'] + '/'
    else:
        resultdir = 'results_batch/' + problem["method"] + "_" + \
                                       problem['identifier'] + '/'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    with open(resultdir + 'log.txt', "w") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
        f.write("\n")
    method = problem["method"]
    verbosity = problem["verbosity"]
    sparsity_level = problem["sparsity_level"]
    meta_results = np.zeros((9, problem['num_tests']))
    np.random.seed(problem["random_seed"])
    problem_type = problem["problem_type"]
    solver_parameter = problem.get('solver_parameter', {})
    for i in range(problem['num_tests']):
        if verbosity:
            print "\nRun example {0}/{1}".format(i + 1, problem['num_tests'])
        random_state = np.random.get_state()
        problem["random_state"] = random_state
        # Creating problem data
        if problem_type == "unmixing":
            A, y, u_real, v_real = create_data_unmixing(problem)
            signal_to_signal_noise_ratio = np.linalg.norm(A.dot(u_real))/ \
                                                np.linalg.norm(A.dot(v_real))
        elif problem_type == "pertubation":
            A, y, u_real, E = create_data_pertubation(problem)
            signal_to_signal_noise_ratio = np.linalg.norm(A.dot(u_real))/ \
                                                np.linalg.norm(E.dot(u_real))
        else:
            raise RuntimeError("Problem type {0} not recognized. Available {1}".format(
                problem_type, __available_problem_types__))
        target_support = np.where(u_real)[0]
        if not os.path.exists(resultdir + str(i) + "_data.npz"):
            success, support, target_support, elapsed_time, relative_error = \
                                recover_support(A, y, u_real, method,
                                                sparsity_level,
                                                verbose=verbosity,
                                                **solver_parameter)
            symmetric_diff = symmetric_support_difference(support,
                                                          target_support)
            symmetric_diff_fixed_size = symmetric_diff + (len(target_support) - \
                                                            len(support))
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                elapsed_time=elapsed_time,
                                symmetric_difference=symmetric_diff,
                                symmetric_diff_fixed_size=symmetric_diff_fixed_size,
                                support=support,
                                success=success,
                                relative_error=relative_error,
                                ssnr=signal_to_signal_noise_ratio)
        else:
            # FIXME: Only for repairing symmetric differences
            datafile = np.load(resultdir + str(i) + "_data.npz")
            elapsed_time=datafile['elapsed_time']
            symmetric_difference=datafile['symmetric_difference']
            support=datafile['support']
            success=datafile['success']
            relative_error=datafile['relative_error']
            ssnr=signal_to_signal_noise_ratio
            symmetric_diff_fixed_size= symmetric_difference + (len(target_support) - \
                                                            len(support))
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                elapsed_time=elapsed_time,
                                symmetric_difference=symmetric_difference,
                                symmetric_diff_fixed_size=symmetric_diff_fixed_size,
                                support=support,
                                success=success,
                                relative_error=relative_error,
                                ssnr=signal_to_signal_noise_ratio)
    create_meta_results(resultdir)
    if verbosity:
        print_meta_results(resultdir)
    else:
        print "Finished simulations."


def create_meta_results(folder):
    """ Method analyses the results of a batch of runs for a single
    constellations. The files that correspond to these runs should be
    contained in the given folder and be named as

        folder + <num_run> + _data.npz.

    where num_run runs from 0 to the number of runs. The data files should
    contain the information as saved for example in the
    'run_numerous_one_constellation' method. The meta results consist of

    "success" : Correct support was identified.
    "symmetric_difference" : Symmetric difference.
    "elapsed_time" : Elapsed time
    "relative_error" : Relative error.

    They are stored in the given folder and named as meta.npz.

    Parameters
    ------------
    folder : string
        Foldername in which files <num_run> + _data.npz are stored.
    """
    success = []
    symmetric_difference = []
    symmetric_diff_fixed_size = []
    elapsed_time = []
    relative_error = []
    ssnr = []
    meta_results_tmp = np.zeros(4)
    i = 0
    while os.path.exists(folder + str(i) + "_data.npz"):
        datafile = np.load(folder + str(i) + "_data.npz")
        success.append(datafile['success'])
        symmetric_difference.append(datafile['symmetric_difference'])
        symmetric_diff_fixed_size.append(datafile['symmetric_diff_fixed_size'])
        elapsed_time.append(datafile['elapsed_time'])
        relative_error.append(datafile['relative_error'])
        ssnr.append(datafile['ssnr'])
        i += 1
    np.savez_compressed(folder + "meta",
                        elapsed_time=np.array(elapsed_time),
                        symmetric_difference=np.array(symmetric_difference),
                        symmetric_diff_fixed_size=np.array(symmetric_diff_fixed_size),
                        success=np.array(success),
                        relative_error=np.array(relative_error),
                        ssnr=ssnr)

def print_meta_results(folder):
    """ Method to print out the meta results to the terminal. The print-out
    shows:
    1) Percentages of successful cases, in which the recovered support is
        correct.
    2) Statistics about the time for specific runs (mean, variance, min, max).
    3) Failed cases with symmetric difference.

    Parameters
    ------------
    folder: string
        Foldername of where to the respective find 'meta.txt' file. Note that
        the full searched location is given by pwd+'<folder>/meta.txt'.
    """
    meta_results = np.load(folder + "/meta.npz")
    num_tests = meta_results["success"].shape[0]
    print "================== META RESULTS ======================"
    print "1) Percentages:"
    print "Support at the end recovered: {0}".format(
                np.sum(meta_results['success'])/float(num_tests))
    print "\n2) Timings:"
    print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
        np.mean(meta_results["elapsed_time"]),
        np.var(meta_results["elapsed_time"]),
        [np.min(np.percentile(meta_results["elapsed_time"], 0.95)),
            np.max(np.percentile(meta_results["elapsed_time"], 95))],
        np.min(meta_results["elapsed_time"]),
        np.max(meta_results["elapsed_time"]))
    print "\n3) Suspicious cases:"
    incorrect_supp = np.where(meta_results["success"] == 0)[0]
    print "Examples support not correct: {0}".format(incorrect_supp)
    print "Symmetric differences unequal to zero: {0}".format(
                zip(incorrect_supp, meta_results["symmetric_difference"]
                                                [incorrect_supp]))
