#!/usr/bin/env python3
"""
This module have a class MultiNormal that represents
 a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
    The class represents a Multivariate Normal distribution
    """

    def __init__(self, data):

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1, keepdims=True)

        covariance = ((data - mean) @ (data - mean).T) / (n - 1)

        self.mean = mean
        self.cov = covariance
