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
        """
        The constructor for finding the mean and covariance.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1, keepdims=True)

        covariance = ((data - mean) @ (data - mean).T) / (n - 1)

        self.mean = mean
        self.cov = covariance

    def pdf(self, x):
        """
        The function to calculates the PDF
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, n = self.mean.shape

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        inverse_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_factor = 1.0 / (np.sqrt((2 * np.pi) ** d * det_cov))

        exponent = -0.5 * ((x - self.mean).T @ inverse_cov @ (x - self.mean))

        pdf = norm_factor * np.exp(exponent).item()

        return pdf
