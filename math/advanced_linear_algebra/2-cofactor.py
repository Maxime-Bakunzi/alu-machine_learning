#!/usr/bin/env python3
"""The module of Function that calculates the cofactor matrix of a matrix"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) -
                (matrix[0][1] * matrix[1][0]))
    det = []
    for i in range(len(matrix)):
        mini = [[j for j in matrix[i]] for i in range(1, len(matrix))]
        for j in range(len(mini)):
            mini[j].pop(i)
        if i % 2 == 0:
            det.append(matrix[0][i] * determinant(mini))
        if i % 2 == 1:
            det.append(-1 * matrix[0][i] * determinant(mini))
    return sum(det)


def cofactor(matrix):
    """Function that calculates the cofactor matrix of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if len(matrix) != len(i):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]
    size = len(matrix)
    if size == 2:
        cofactor_matrix = [[matrix[1][1], -matrix[1][0]],
                           [-matrix[0][1], matrix[0][0]]]
        return cofactor_matrix
    cofactor_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            minor_matrix = [row[:j] + row[j + 1:]
                            for row in (matrix[:i] + matrix[i + 1:])]
            sign = (-1) ** (i + j)
            cofactor_matrix[i][j] = sign * determinant(minor_matrix)
    return cofactor_matrix
