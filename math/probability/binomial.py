#!/usr/bin/env python3
"""
Binomial distribution class module
"""


from math import comb

class Binomial:
    """
    Represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution

        Args:
            data (list, optional): list of the data to be used to
             estimate the distribution
            n (int): number of Bernoulli trials
            p (float): probability of a “success”

        Raises:
            ValueError: If n is not positive or if p is not a valid probability
            TypeError: If data is not a list
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n