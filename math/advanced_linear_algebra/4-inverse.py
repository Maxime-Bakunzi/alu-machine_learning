#!/usr/bin/env python3
"""Function that calculates the inverse of a matrix"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] -
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
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]
    if len(matrix) == 2:
        cofactor_matrix = [[matrix[1][1], -matrix[1][0]],
                           [-matrix[0][1], matrix[0][0]]]
        return cofactor_matrix
    cofactor_matrix = [[0] * len(matrix) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            minor = [row[:j] + row[j + 1:]
                     for row in (matrix[:i] + matrix[i + 1:])]
            sign = (-1) ** (i + j)
            cofactor_matrix[i][j] = sign * determinant(minor)
    return cofactor_matrix


def adjugate(matrix):
    """Function that calculates the adjugate matrix of a matrix"""
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [
        [cofactor_matrix[j][i] for j in range(len(cofactor_matrix))]
        for i in range(len(cofactor_matrix))
    ]
    return adjugate_matrix


def inverse(matrix):
    """Function that calculates the inverse of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")
    det = determinant(matrix)
    if det == 0:
        return None
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1 / det]]
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = [[element / det for element in row]
                      for row in adjugate_matrix]
    return inverse_matrix
