# coding: utf8
import getopt
import json
import os
import sys

import numpy as np

from encoders.handler import recover_support, check_method_validity
from problem_factory.synthetic_random_data import \
    create_specific_problem_data_from_problem
from calculation_utilities.general import symmetric_support_difference


def run_single(problem):
    """ Create tiling of a problem of type

        A * (u + v) = y + eps

    with randomly created data A, u, v and eps. The run characteristics (ie.
    noise levels, noise types, signal and noise strength and so forth) are
    given in the dictionary called 'problem'. Also, the
    dictionary stores other important characteristics of the run. Concretely,
    the dictonary must contain the following information:

    Dictionary-key | Description
    ---------------------------
    identifier | Subfolder identifier where the results shall be stored. Full
                 path will be "/results/<method>_<identifier>/".
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

    Method will save the results to a file called data.npz
    in the folder 'results_single/<method>_<identifier>/'.
    If the file already exists, the file will be overwritten. Therefore, be
    careful if running several times with equal identifier.
    """
    resultdir = 'results_single/' + problem['method'] + '_' + \
                                    problem['identifier'] + '/'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    with open(resultdir + 'log.txt', "a+") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
    # Extract encoder from problem data
    method = problem["method"]
    sparsity_level = problem["sparsity_level"]
    # Extract random stuff
    np.random.seed(problem["random_seed"])
    random_state = np.random.get_state()
    problem["random_state"] = random_state
    # Creating problem data
    A, y, u_real, v_real = create_specific_problem_data_from_problem(problem)
    success, support, target_support, elapsed_time, relative_error = \
                                recover_support(A, y, u_real, v_real, method,
                                                sparsity_level, verbose=True)
    symmetric_diff = symmetric_support_difference(support, target_support)
    np.savez_compressed(resultdir + "data.npz",
                        elapsed_time=elapsed_time,
                        symmetric_difference=symmetric_diff,
                        support=support,
                        success=success,
                        relative_error=relative_error)

def main(argv):
    identifier = ''
    method = ''
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_single.py -i <identifier> -m "
               "<method>.\n"
               "<identifier> is an arbitraray folder name.\n"
               "<method> specifies the sparse encoder to use.\n"
               "The run characteristics are specified inside 'run_single.py' file.\n"
               "===============================================================\n")
    try:
        opts, args = getopt.getopt(argv, "i:m:h", ["identifier=", "method=",
                                                     "help"])
    except getopt.GetoptError:
        print helpstr
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print helpstr
            sys.exit()
        elif opt in ("-i", "--identifier"):
            identifier = arg
        elif opt in ("-m", "--method"):
            method = arg
    if identifier == '':
        print "Please add identifer. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    if method == '' or not check_method_validity(method):
        print "Please add valid method. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    print "Running single simulation. Results will be stored in folder {0}".format(
        identifier)
    problem = {
        'identifier': identifier,
        'method': method,
        'n_measurements': 250,
        'n_features': 1250,
        'sparsity_level': 10,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform_ensured_max',
        'noise_lev_signal': 0.2,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 123456
    }
    run_single(problem)


if __name__ == "__main__":
    main(sys.argv[1:])
