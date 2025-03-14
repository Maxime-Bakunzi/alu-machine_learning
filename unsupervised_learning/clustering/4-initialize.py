#!/usr/bin/env python3
"""
Defines function that initializes variables for a Gaussian Mixture Model
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            containing the number of clusters

    not allowed to use any loops

    returns:
        pi, m, S:
            pi [numpy.ndarray of shape (k,)]:
                containing the priors for each cluster, initialized evenly
            m [numpy.ndarray of shape (k, d)]:
                containing the centroid means for each cluster,
                    initialized with K-means
            S [numpy.ndarray of shape (k, d, d)]:
                containing the covariance matrices for each cluster,
                    initialized as identity matrices
        or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    C, clss = kmeans(X, k)
    pi = np.full(k, 1 / k)
    m = C
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
