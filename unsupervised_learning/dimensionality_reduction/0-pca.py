#!/usr/bin/env python3
"""
Defines a function that performs principal Component Analysis (PCA)
 on a dataset.
"""

import numpy as np


def pca(X, var=0.95):
    """
    performs Principal Compenent Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) with zero mean across
          all dimensions.
        var (float, optional): Fraction of varience to maintain.
          Defaults to 0.95.

    Returns:
        nump.ndarray: Weight matrix W of shape (d, nd)
         where nd is the new dimensionality.
    """
    # Center the data by subtracting the mean of each feature
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Calculate the explained varience ratios
    explained_var = S ** 2
    total_var = np.sum(explained_var)
    explained_var_ratios = explained_var / total_var

    # Compute cumulative explained varience and find the number of components
    cumulative_var = np.cumsum(explained_var_ratios)
    nd = np.argmax(cumulative_var >= var) + 1

    # Construct the weight matrix from the top nd components
    W = Vt[:nd].T
    T = np.dot(X_centered, W)

    return T
