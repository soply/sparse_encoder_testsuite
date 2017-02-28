# coding: utf8
""" Callable run script to perform single experiments. """

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
        See problem_factory/synthetic_random_data docs for details on the run
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
    problem = {
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
    main(sys.argv[1:], problem)
