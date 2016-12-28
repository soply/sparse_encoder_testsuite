import numpy as np

from least_squares_regression import least_squares_regression


def approximate_unmixing_solution(matrix, rhs, support):
    """ doc """
    u_I = least_squares_regression(matrix, rhs, support = support)
    rhs = rhs - matrix[:,support].dot(u_I[support])
    v_I = least_squares_regression(matrix, rhs)
    return u_I, v_I
