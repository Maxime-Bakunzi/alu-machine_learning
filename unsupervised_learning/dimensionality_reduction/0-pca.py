#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset with zero mean.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) with zero mean.
        var (float): Fraction of variance to retain. Defaults to 0.95.

    Returns:
        numpy.ndarray: Weights matrix of shape (d, nd).
    """
    # Perform SVD on the already-centered X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Calculate explained variance ratios
    explained_var = S ** 2
    total_var = np.sum(explained_var)
    explained_var_ratios = explained_var / total_var

    # Find the number of components to retain 'var' variance
    cumulative_var = np.cumsum(explained_var_ratios)
    nd = np.argmax(cumulative_var >= var) + 1

    # Return the weights matrix (principal components)
    W = Vt[:nd].T
    return W
