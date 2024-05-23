#!/usr/bin/env python3
"""Function that calculates the adjugate matrix of a matrix"""


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
    if len(matrix) == 2:
        cofactor_matrix = [i[::-1] for i in matrix][::-1]
        cofactor_matrix = [[cofactor_matrix[i][j] if (i + j) % 2 == 0 else
                            -cofactor_matrix[i][j]
                            for j in range(len(cofactor_matrix[i]))]
                           for i in range(len(cofactor_matrix))]
        return cofactor_matrix
    cofactor_matrix = [[j for j in matrix[i]] for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            minor = [[j for j in matrix[i]] for i in range(len(matrix))]
            minor = minor[:i] + minor[i + 1:]
            for k in range(len(minor)):
                minor[k].pop(j)
            if (i + j) % 2 == 0:
                cofactor_matrix[i][j] = determinant(minor)
            if (i + j) % 2 == 1:
                cofactor_matrix[i][j] = -1 * determinant(minor)
    return cofactor_matrix


def adjugate(matrix):
    """Function that calculates the adjugate matrix of a matrix"""
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
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [[cofactor_matrix[j][i]
                       for j in range(len(cofactor_matrix))]
                       for i in range(len(cofactor_matrix))]
    return adjugate_matrix
