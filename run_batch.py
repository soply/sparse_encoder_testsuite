# coding: utf8
""" Methods to run and analyse repitions of experiments with synthetic random
    data. """

__author__ = "Timo Klock"

import getopt
import json
import os
import sys

import numpy as np

from calculation_utilities.general import symmetric_support_difference
from encoders.handler import check_method_validity, recover_support
from problem_factory.synthetic_random_data import \
    create_specific_problem_data_from_problem


def run_numerous_one_constellation(problem, results_prefix = None):
    """ Create numerous simulations of problems of type

        A * (u + v) = y + eps

    with randomly created data A, u, v and eps. The run characteristics (ie.
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
    noise_type_signal | Type of noise that is applied to the signal (ie. type
                        of noise of v).
    noise_lev_signal | Noise level of the signal noise.
    noise_type_measurements | Type of noise that is applied to the measurements
                              y (ie. type of noise of eps).
    noise_lev_measurements | Noise level of the measurement noise.
    random_seed | Random seed for the data creation. If given and fixed, the
                  same random data is created.
    verbosity | If false, output will be very minimized.

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
    for i in range(problem['num_tests']):
        if verbosity:
            print "\nRun example {0}/{1}".format(i + 1, problem['num_tests'])
        random_state = np.random.get_state()
        problem["random_state"] = random_state
        # Creating problem data
        A, y, u_real, v_real = create_specific_problem_data_from_problem(
            problem)
        target_support = np.where(u_real)[0]
        if not os.path.exists(resultdir + str(i) + "_data.npz"):
            success, support, target_support, elapsed_time, relative_error = \
                                recover_support(A, y, u_real, v_real, method,
                                                sparsity_level,
                                                verbose=verbosity)
            symmetric_diff = symmetric_support_difference(support, target_support)
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                elapsed_time=elapsed_time,
                                symmetric_difference=symmetric_diff,
                                support=support,
                                success=success,
                                relative_error=relative_error)
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
    elapsed_time = []
    relative_error = []
    meta_results_tmp = np.zeros(4)
    i = 0
    while os.path.exists(folder + str(i) + "_data.npz"):
        datafile = np.load(folder + str(i) + "_data.npz")
        success.append(datafile['success'])
        symmetric_difference.append(datafile['symmetric_difference'])
        elapsed_time.append(datafile['elapsed_time'])
        relative_error.append(datafile['relative_error'])
        i += 1
    np.savez_compressed(folder + "meta",
                        elapsed_time=np.array(elapsed_time),
                        symmetric_difference=np.array(symmetric_difference),
                        success=np.array(success),
                        relative_error=np.array(relative_error))

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


def main(argv, problem):
    """ Method to run batch of problems. Can be used from terminal line (run
    characteristics specified below) or as a function.

    Parameters
    -------------
    argv : python list with 6 options and arguments to run simulation.
        Example: argv = ['t', 'run', 'i', 'test', 'm', 'omp']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/synthetic_random_data docs for details on the run
        characteristics.
    """
    identifier = ''
    method = ''
    task = ''
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_batch.py -t <task> " +\
               "-i <identifier>' -m <method> .\n"
               "<task> can be 'run' to simula a new batch or 'show' to show\n"
               "results of a previous run. \n"
               "<identifier> is an arbitrary folder name.\n"
               "<method> specifies the sparse encoder to use.\n"
               "The run characteristics are specified inside 'run_batch.py' file.\n"
               "===============================================================\n")
    try:
        opts, args = getopt.getopt(argv, "t:i:m:h", ["task=", "identifier=",
                                                     "method=", "help"])
    except getopt.GetoptError:
        print helpstr
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpstr
            sys.exit()
        elif opt in ("-i", "--identifier"):
            identifier = arg
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-m", "--method"):
            method = arg
    if identifier == '':
        print "Please add identifer. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    if method == '' or not check_method_validity(method, verbose = True):
        print "Please add valid method. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    problem.update({'identifier': identifier, 'method' : method})
    if task == 'run':
        print "Running batch simulation. Results will be stored in folder {0}".format(
            identifier)
        run_numerous_one_constellation(problem)
    elif task == 'show':
        try:
            print_meta_results('results_batch/' + method + "_" + \
                               identifier + '/')
        except IOError:
            print ("Could not load specified file. Check folder  "
                    "'results_batch/{0}/' for meta file please.'".format(
                        method + "_" + identifier))
        finally:
            sys.exit(2)
    else:
        print "Please specify task task. Run file as follows:\n"
        print helpstr
        sys.exit(2)

if __name__ == "__main__":
    problem = {
        'num_tests': 100,
        'n_measurements': 250,
        'n_features': 800,
        'sparsity_level': 15,
        'smallest_signal': 1.5,
        'largest_signal': 50.0,
        'noise_type_signal': 'uniform_ensured_max',
        'noise_lev_signal': 0.2,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1223445,
        'verbosity' : False
    }
    main(sys.argv[1:], problem)
