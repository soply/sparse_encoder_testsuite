# coding: utf8
""" Callable run script to perform single experiments. """

__author__ = "Timo Klock"

import getopt
import sys

from sparse_encoder_test_suite.run_single import run_single
from sparse_encoder_test_suite.encoders.handler import check_method_validity

def main(argv, problem):
    """ Method to run single problem. Can be used from terminal line (run
    characteristics specified below) or as a function.

    Parameters
    -------------
    argv : python list with 4 options and arguments to run simulation.
        Example: argv = ['i', 'test', 'm', 'omp']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/ docs for details on the run
        characteristics.
    """
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
    problem.update({'identifier': identifier, 'method' : method})
    print "Running single simulation. Results will be stored in folder {0}".format(
        identifier)
    run_single(problem)


if __name__ == "__main__":
    solver_parameter = {
        'beta_min' : 1e-6,
        'beta_max' : 100.0,
        'n_beta' : 10,
        'beta_scaling' : 'logscale',
        'suppress_warning' : True,
    }
    problem = {
        'n_measurements': 250, # m
        'n_features': 1250, # n
        'sparsity_level': 6, # Sparsity level of u
        'smallest_signal': 1.5, # Smallest signal size in u
        'largest_signal': 2.0, # Largest signal size in u
        'noise_type_signal': 'uniform_ensured_max', # Uniformly distributed noise where the maximum allowed value is taken for sure
        'noise_lev_signal': 0.2, # Noise level of the vector v (exact meaning depends on noise type)
        'noise_type_measurements': 'gaussian', # Additional measurement noise if desired. Can be of the same type.
        'noise_lev_measurements': 0.0, # Noise level of the additional measurement noise.
        'random_seed': 12, # Just to fix the randomness
        'sampling_matrix_type' : 'prtm_rademacher', # Partial random circulant matrix from a Rademacher sequence
        'problem_type' : 'unmixing' # 'unmixing' A(u+v) = y or 'pertubation' (A + E)u = y
    }
    main(sys.argv[1:], problem)
