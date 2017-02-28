# coding: utf8
""" Callable run script to perform numerous simulations with multiple
    constellations. """

__author__ = "Timo Klock"

import getopt
import sys
import os

from sparse_encoder_test_suite.encoders.handler import check_method_validity
from sparse_encoder_test_suite.run_multiple_batches import (print_meta_results,
                                        run_numerous_multiple_constellations)


def main(argv, problem):
    """ Method to run multiple batches of problems. Can be used from terminal
    line (run characteristics specified below) or as a function.

    Parameters
    -------------
    argv : python list with 6 options and arguments to run simulation.
        Example: argv = ['t', 'run', 'i', 'test', 'm', 'omp']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/synthetic_random_data docs for details on the run
        characteristics.
    """
    identifier = ''
    task = ''
    method = ''
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_multiple_batchs.py -t <task> "
               "-i <identifier> -m <method>'.\n"
               "<task> can be 'run' to simula new batches or 'show' to show\n"
               "results of all runs belonging to a previous batch. \n"
               "<identifier> is an arbitrary folder name.\n"
               "<method> specifies the sparse encoder to use.\n"
               "The run characteristics are specified inside"
               " 'run_multiple_batches.py' file.\n"
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
        print ("Running multiple batch simulation. Results will be stored in"
               " subfolders of '{0}'.".format('results_multiple_batches/' +
                                              identifier + '/'))
        run_numerous_multiple_constellations(problem)
    elif task == 'show':
        ctr = 0
        resultsdir = 'results_multiple_batches/' + method + "_" + identifier + '/'
        print "\n================= Meta results of all runs ================="
        while os.path.exists(resultsdir + str(ctr) + "/0_data.npz"):
            print "\nRun {0}".format(ctr)
            print_meta_results(resultsdir + str(ctr) + "/")
            ctr += 1
        else:
            print ("\n\nFound {0} runs for identifier '{1}' and "
                   "basedirectory '{2}'.".format(str(ctr), identifier, resultsdir))
    else:
        print "Please specify task task. Run file as follows:\n"
        print helpstr
        sys.exit(2)

if __name__ == "__main__":
    problem = {
        'num_tests': 20,
        'n_measurements': [350, 500, 750],
        'n_features': [1250, 1250, 1500],
        'sparsity_level': 8,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform_ensured_max',
        'noise_lev_signal': 0.3,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1,
        'verbosity' : False
    }
    main(sys.argv[1:], problem)
