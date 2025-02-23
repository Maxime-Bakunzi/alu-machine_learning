#!/usr/bin/env python3
"""
Defines a function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        kmin [positive int]:
            the minimum number of clusters to check for (inclusive)
        kmax [positive int]:
            the maximum number of clusters to check for (inclusive)
            if None, kmax should be set to maximum number of clusters possible
        iterations [positive int]:
            the maximum number of iterations for the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used for early stopping
        verbose [boolean]:
            determines if you should print information about the algorithm

    should only use one loop

    returns:
        best_k, best_result, l, b
            best_k [positive int]:
                the best value for k based on its BIC
            best_result [tuple containing pi, m, S]:
                pi [numpy.ndarray of shape (k,)]:
                    contains cluster priors for the best number of clusters
                m [numpy.ndarray of shape (k, d)]:
                    contains centroid means for the best number of clusters
                S [numpy.ndarray of shape (k, d, d)]:
                    contains covariance matrices for best number of clusters
            l [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the log likelihood for each cluster size tested
            b [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the BIC value for each cluster size tested
                BIC = p * ln(n) - 2 * 1
                    p: number of parameters required for the model
                    n: number of data points used to create the model
                    l: the log likelihood of the model
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S,  _, log_l = expectation_maximization(X, k, iterations, tol,
                                                       verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        # code based on gaussian mixture source code n_parameters source code
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        # Formula for this task BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    k_best = k_best[best_val]
    best_res = best_res[best_val]

    return k_best, best_res, logl_val, bic_val
