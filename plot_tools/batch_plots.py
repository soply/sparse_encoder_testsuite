# coding: utf8

import matplotlib.pyplot as plt
import numpy as np


def plot_success_wrt_sparsity_param_matrixdimensions(x_axis, y_axis,
                                                     identifier_list,
                                                     plot_title,
                                                     legend_entries = None):
    


    # success_percentages = np.zeros((len(s), len(matdim)))
    # success_percentages2 = np.zeros((len(s), len(matdim)))
    # success_percentages3 = np.zeros((len(s), len(matdim)))
    # success_percentages4 = np.zeros((len(s), len(matdim)))
    # plt.figure()
    # plt.axis([1, 24, 0, 1.1])
    # identifier = "visualize_folding_realwithout"
    # identifier2 = "visualize_folding_without"
    # identifier3 = "results_trondheim"
    # identifier4 = "results_trondheim_unfolded"
    # colors = ['r','b','g','c','y']
    # for i in range(1,len(matdim)):
    #     for j in range(len(s)):
    #         meta_results = np.load("results/" + identifier + "_" + str(j) + "_" + \
    #             str(i) + "_" + method + "/" + "meta.npy")
    #         meta_results2 = np.load("results/" + identifier2 + "_" + str(j) + "_" + \
    #             str(i) + "_" + method + "/" + "meta.npy")
    #         try:
    #             meta_results3 = np.loadtxt("results/" + identifier3 + "_" + str(j) + "_" + \
    #                 str(i) + "/" + "meta.gz")
    #         except IOError:
    #             meta_results3 = np.ones((9, 100))
    #         try:
    #             meta_results4 = np.loadtxt("results/" + identifier4 + "_" + str(j) + "_" + \
    #                 str(i) + "/" + "meta.gz")
    #         except IOError:
    #             meta_results4 = np.ones((9, 100))
    #         success_percentages3[j,i] = len(np.where(meta_results3[5,:] == 0)[0]) /  float(len(meta_results3[5,:]))
    #         success_percentages[j,i] = np.sum(meta_results[0,:]) / float(len(meta_results[0,:]))
    #         success_percentages2[j,i] = np.sum(meta_results2[0,:]) / float(len(meta_results2[0,:]))
    #         success_percentages4[j,i] = np.sum(meta_results4[5,:] == 0) / float(len(meta_results4[5,:] == 0))
    #     plt.plot(s, success_percentages[:,i], (colors[i] + "-o"), linewidth = 4,
    #         label = "{0} x {1}".format(matdim[i][0], matdim[i][1]))
    #     plt.plot(s, success_percentages2[:,i], (colors[i]+"--"),  linewidth = 4,
    #         label = "{0} x {1} NF".format(matdim[i][0], matdim[i][1]))
    #     plt.plot(s, success_percentages4[:,i], (colors[i]+":"), linewidth = 4,
    #         label = "{0} x {1} MP".format(matdim[i][0], matdim[i][1]))
    #     plt.plot(s, success_percentages3[:,i], (colors[i]+"-."),  linewidth = 4,
    #         label = "{0} x {1} MP NF".format(matdim[i][0], matdim[i][1]))
    #
    # plt.xlabel('Sparsity')
    # plt.ylabel('Successful recovery [%]')
    # plt.legend(loc = "lower left")
    # plt.title("Effect of noise folding with original SNR = {1}".format(method, snr))
    # plt.show()
