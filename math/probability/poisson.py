#!/usr/bin/env python3
"""
Poisson distribution class module
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution

        Args:
            data : list of the data to estimate the distribution
            lambtha : expected number of occurrences in a given time frame

        Raises:
            ValueError: If lambtha is not positive or has less than two values
            TypeError: If data is not a list
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    @staticmethod
    def factorial(k):
        """
        Computes the factorial of k

        Args:
            k (int): The value to compute the factorial of

        Returns:
            int: The factorial of k
        """
        if k == 0:
            return 1
        else:
            return k * Poisson.factorial(k - 1)
