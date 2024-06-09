#!/usr/bin/env python3
"""
Binomial class which represents a binomial distribution.
"""


class Binomial:
    """
    Represents a binomial distribution.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution with data, n, and p.

        Parameters:
        data (list): A list of data points to estimate the distribution.
        n (int): The number of Bernoulli trials.
        p (float): The probability of a "success".
        """
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not isinstance(p, (float, int)) or not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_estimate = 1 - (variance / mean)
            n_estimate = round(mean / p_estimate)
            self.n = n_estimate
            self.p = mean / self.n
