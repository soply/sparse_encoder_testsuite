# coding: utf8
""" Callable run script to perform batch experiments"""

__author__ = "Timo Klock"

import getopt
import sys

from sparse_encoder_test_suite.encoders.handler import check_method_validity
from sparse_encoder_test_suite.run_batch import (print_meta_results,
                                                 run_numerous_one_constellation)


def main(argv, problem):
    """ Method to run batch of problems. Can be used from terminal line (run
    characteristics specified below) or as a function.

    Parameters
    -------------
    argv : python list with 6 options and arguments to run simulation.
        Example: argv = ['t', 'run', 'i', 'test', 'm', 'omp']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/ docs for details on the run
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
        'verbosity' : False,
        'sampling_matrix_type' : 'prtm_gaussian'
    }
    main(sys.argv[1:], problem)
