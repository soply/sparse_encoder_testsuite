#coding: utf8
from tabulate import tabulate

from main import recover_support
from problem_factory import create_specific_problem_data

import matplotlib.pyplot as plt
import numpy as np

def measure_wrt_measurements_pb_sparsitylevel(
        M,
        L,
        method,
        N = 1000,
        c = 1.5,
        d = 0.1,
        random_seed = 12345,
        noise_type = "linf_bounded",
        repititions = 100,
        check_conditions = False,
        verbose = False):
    # Allocate memory for timings
    timing = np.zeros((len(L),len(M)))
    success_percentage = np.zeros((len(L), len(M)))
    # figure
    plt.figure()
    np.random.seed(random_seed)
    for j in range(len(L)):
        for i in range(len(M)):
            for _ in range(repititions):
                random_state = np.random.get_state()
                A, y, u, v = create_specific_problem_data(M[i], N, L[j], c, d,
                        random_seed = random_seed,
                        random_state = random_state,
                        noise_type = noise_type)
                # Run problem
                success, support, target_support, elapsed_time, relative_error = \
                    recover_support(A, y, u, method, L[j], verbose = verbose)
                timing[j,i] += elapsed_time
                if success == 1:
                    success_percentage[j,i] += 1
            success_percentage[j,i] = success_percentage[j,i]/float(repititions)
            timing[j,i] = timing[j,i]/float(repititions)
        plt.plot(M, timing[j,:], '-o', label = "s={0}".format(L[j]))
    # Tabulate success rates
    success_percentage = np.vstack((M, success_percentage)).T
    header = ["Measurements"]
    for j in range(len(L)):
        header.append("Succ. s = {0}".format(L[j]))
    print tabulate(success_percentage, headers = header)
    # Plot timings
    plt.xlabel('Measurements')
    plt.ylabel('Elapsed time [s]')
    plt.title('Timing w.r.t. measurements (param. by sparsity)')
    plt.legend(loc = "lower right")
    plt.show()

def measure_wrt_features_pb_sparsitylevel(
        N,
        L,
        method,
        M = 240,
        c = 1.5,
        d = 0.1,
        random_seed = 12345,
        noise_type = "linf_bounded",
        repititions = 100,
        check_conditions = False,
        verbose = False):
    # Allocate memory for timings
    timing = np.zeros((len(L),len(N)))
    success_percentage = np.zeros((len(L), len(N)))
    # figure
    plt.figure()
    np.random.seed(random_seed)
    for j in range(len(L)):
        for i in range(len(N)):
            for _ in range(repititions):
                random_state = np.random.get_state()
                A, y, u, v = create_specific_problem_data(M, N[i], L[j], c, d,
                        random_seed = random_seed,
                        random_state = random_state,
                        noise_type = noise_type)
                # Run problem
                success, support, target_support, elapsed_time, relative_error = \
                    recover_support(A, y, u, method, L[j], verbose = verbose)
                timing[j,i] += elapsed_time
                if success == 1:
                    success_percentage[j,i] += 1
            success_percentage[j,i] = success_percentage[j,i]/float(repititions)
            timing[j,i] = timing[j,i]/float(repititions)
        plt.plot(N, timing[j,:], '-o', label = "s={0}".format(L[j]))
    # Tabulate success rates
    success_percentage = np.vstack((N, success_percentage)).T
    header = ["Features"]
    for j in range(len(L)):
        header.append("Succ. s = {0}".format(L[j]))
    print tabulate(success_percentage, headers = header)
    # Plot timings
    plt.xlabel('Features')
    plt.ylabel('Elapsed time [s]')
    plt.title('Timing w.r.t. features (param. by sparsity)')
    plt.legend(loc = "lower right")
    plt.show()

def measure_wrt_sparsity_pb_matrix_dimensions(
        L,
        matrix_dimensions,
        method,
        c = 1.5,
        d = 0.1,
        random_seed = 12345,
        noise_type = "linf_bounded",
        repititions = 100,
        check_conditions = False,
        verbose = False):
    dim1 = len(matrix_dimensions)
    dim2 = len(L)
    # Allocate memory for timings
    timing = np.zeros((dim1,dim2))
    success_percentage = np.zeros((dim1,dim2))
    # figure
    plt.figure()
    np.random.seed(random_seed)
    for j in range(dim1):
        for i in range(dim2):
            for _ in range(repititions):
                random_state = np.random.get_state()
                A, y, u, v = create_specific_problem_data(matrix_dimensions[j][0],
                    matrix_dimensions[j][1], L[i], c, d,
                    random_seed = random_seed,
                    random_state = random_state,
                    noise_type = noise_type)
                # Run problem
                success, support, target_support, elapsed_time, relative_error = \
                    recover_support(A, y, u, method, L[i], verbose = verbose)
                timing[j,i] += elapsed_time
                if success == 1:
                    success_percentage[j,i] += 1
            success_percentage[j,i] = success_percentage[j,i]/float(repititions)
            timing[j,i] = timing[j,i]/float(repititions)
        plt.plot(L, timing[j,:], '-o', label = "Mat. dim = {0} x {1}".format(
            matrix_dimensions[j][0], matrix_dimensions[j][1]))
    # Tabulate success rates
    success_percentage = np.vstack((L, success_percentage)).T
    header = ["Matrix dimensions"]
    for j in range(len(matrix_dimensions)):
        header.append("{0} x {1}".format(matrix_dimensions[j][0], matrix_dimensions[j][1]))
    print tabulate(success_percentage, headers = header)
    # Plot timings
    plt.xlabel('Sparsity level')
    plt.ylabel('Elapsed time [s]')
    plt.title('Timing w.r.t. sparsity (param. by matrix dimension)')
    plt.legend(loc = "lower right")
    plt.show()

if __name__ == "__main__":
    # matrix_dimensions = [[250, 256], [400, 1024]]
    # N = [256, 350, 450, 528, 720, 900, 1056, 1280, 1580, 1800, 2112]
    # L = [4, 6, 12, ]
    # # measure_wrt_features_pb_sparsitylevel(N, L,
    # #     repititions = 140, mpsr_verbosity = 0)
    # M = [56, 80, 128, 160, 200, 256, 325, 400, 500]
    # N = 512
    # L = [4, 6, 12]
    # measure_wrt_measurements_pb_sparsitylevel(M, L, repititions = 100, mpsr_verbosity = 0,
    #     random_seed = 2)
    L = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18]
    matrix_dimensions = [[250, 256], [400, 1024]]
    measure_wrt_sparsity_pb_matrix_dimensions(L, matrix_dimensions, 'lasso',
        repititions = 100, random_seed = 1)
