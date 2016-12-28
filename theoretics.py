#coding: utf8
import matplotlib.pyplot as plt
import numpy as np

""" Sources:
[1] Orthogonal Matching Pursuit for Sparse Signal Recovery With Noise (T. Cai,
    L. Wang)
[2] Sparsity and the Lasso (R. Tibshirani, L. Wassermann)
"""


def omp_minimum_signal_strength(sparsity_grid, mutual_incoherences, noise_type):
    """ Plot curve
            sparsity_level -> lower bound signal strength
    for orthogonal matching pusuit. The bounds can be found in [1]."""
    # Define function from Theorem 1 in [1]
    def lower_bound_l2(s, mi):
        result = np.zeros(len(s))
        mask = mi < 1.0/(2.0 * s - 1)
        result[~mask] = 0
        result[mask] = 2.0/(1.0 - (2.0 * s[mask] - 1) * mi)
        return result
    # Define function from Theorem 4 in [1]
    def lower_bound_inf(s, mi):
        mask = mi < 1.0/(2.0 * s - 1)
        result = np.zeros(len(s))
        result[~mask] = 0
        result[mask] = 2.0/(1.0 - (2.0 * s[mask] - 1) * mi) * \
            (1 + np.sqrt(s[mask])/np.sqrt(1.0 - (s[mask] - 1.0) * mi))
        return result
    # Define function from Theorem 1 in [1] (1/n: percentage of failure)
    def lower_bound_gauss(s, mi, n = 20):
        result = np.zeros(len(s))
        mask = mi < 1.0/(2.0 * s - 1)
        result[~mask] = 0
        result[mask] = 2.0 * np.sqrt(n + 2.0 * np.sqrt(n * np.log(n)))/ \
            (1.0 - (2.0 * s[mask] - 1.0) * mi)
        return result

    if noise_type == "l2_bounded":
        fun = lower_bound_l2
    elif noise_type == "linf_bounded":
        fun = lower_bound_inf
    if noise_type == "gaussian":
        fun = lower_bound_gauss

    # Allocate memory for values
    SNR = np.zeros((len(sparsity_grid), len(mutual_incoherences)))
    plt.figure()
    for i in range(len(mutual_incoherences)):
        mi = mutual_incoherences[i]
        SNR [:,i] = fun(sparsity_grid, mi)
        plt.semilogy(sparsity_grid, SNR[:,i], label = 'mi = {0}'.format(
            mi))
    plt.xlabel('Sparsity level')
    plt.ylabel('Signal strength/noise level')
    plt.title('OMP: Minimum SNR for sparse recovery')
    plt.legend(loc = 'lower right')
    plt.show()

if __name__ == "__main__":
    mutual_incoherences = [0.001, 0.005, 0.01, 0.025, 0.05]
    sparsity_grid = np.arange(1, 200)
    noise_type = "linf_bounded"
    omp_minimum_signal_strength(sparsity_grid, mutual_incoherences, noise_type)
