# coding: utf8
""" Methods to run and analyse repitions of experiments with synthetic random
    data. Different run characteristics are possible. """

__author__ = "Timo Klock"

import json
import os

from encoders.handler import check_method_validity
from run_batch import print_meta_results, run_numerous_one_constellation


def run_numerous_multiple_constellations(problem):
    """ This method is similar to 'run_batch.run_numerous_one_constellation'
    with the difference that the problem dictionary (see docs of
    'run_batch.run_numerous_one_constellation' for more details on the dict) can
    be lists instead of single number. This enables to run multiple batches of
    different run characteristics, ie. different sparsity levels, feature sizes
    or measurement sizes.
    Concretely, the dictionary should contain the same keys as decribed in
    'run_batch.run_numerous_one_constellation'. Each entry should either be
    a single object/value, if the respective characteristic is the same for all
    runs; or it should contain a list of different characteristics that should
    be iterated through. It is important to note that the dictionary can not
    contain multiple lists of different sizes because we iterate simoultaneously
    through all lists of the dictionary (if it is a list). Therefore all lists
    in the dictionary have to be equally sized.
    The results of the respective runs are saved to
    'results_multiple_batches/<method> + "_" + <problem['identifier']>/<i>/'
    where <i> is the number of the respective run.

    Example
    ------------
    problem = {
        'identifier': 'test1',
        'method' : 'lasso',
        'num_tests': 20,
        'n_measurements': [350, 500, 750],
        'n_features': [1250, 1250, 1500],
        'sparsity_level': 8,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform',
        'noise_lev_signal': 0.3,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1,
        'verbosity' : False,
        'sampling_matrix_type': 'gaussian'
    }
    'run_numerous_multiple_constellations(problem)'

    runs three seperated batch
    runs, where all characteristics except 'n_measurements' and 'n_features'
    remain the same for all runs. The latter two charactertics will be
    (350, 1250), (500, 1250), (750,1500) for these three runs.

    Remarks
    ----------------
    a) 'method', 'identifier' and 'verbosity' are not allowed to be lists. They
        are equal for all runs.

    b) Please consult the docs of 'run_batch.run_numerous_one_constellation' for
       more information on the problem dictionary.
    """
    parentdir = "results_multiple_batches/" + problem['method'] + "_" + \
                                  problem['identifier']
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)
    # Write log file to parent folder with problem description
    with open(parentdir + '/log.txt', "w") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
        f.write("\n")
    # Check if problem dictionary contains either a single object, or a list of
    # the same size for each dictionary entry.
    listsize = 1
    for key, val in problem.iteritems():
        if isinstance(val, list):
            if listsize == 1:
                # No list found in dict so far. Set dictsize
                listsize = len(val)
            elif len(val) != listsize:
                raise RuntimeError(("The 'problem' dictionary does not contain"
                                    " equally sized lists. Each entry of the dictionary"
                                    " should either be a single entry or a list of size that"
                                    " consides with all lists in the dictionary.\n"
                                    "Problematic (key,val) : ({0},{1})".format(key, val)))
    # Get's i-th element if list or object if not a list
    problemgetter = lambda x, i: x[i] if isinstance(x, list) else x
    for i in range(listsize):
        subproblem = {key: problemgetter(problem[key], i) for key in
                      problem.keys()}
        subproblem['identifier'] += "/{0}/".format(str(i))
        run_numerous_one_constellation(subproblem,
                                       results_prefix="results_multiple_batches/")
