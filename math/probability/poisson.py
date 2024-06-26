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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”

        Args:
            k (int): The number of “successes”

        Returns:
            float: The PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # Using the formula P(X = k) = (e^(-lambtha) * lambtha^k) / k!
        e = 2.7182818285
        lambtha = self.lambtha
        factorial_k = self.factorial(k)

        pmf = (e ** -lambtha) * (lambtha ** k) / factorial_k
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”

        Args:
            k (int): The number of “successes”

        Returns:
            float: The CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # Formula F(k; lambtha) = P(X <= k)=sum(pmf(i) for i in range(k + 1))
        e = 2.7182818285
        lambtha = self.lambtha

        cdf = 0
        for i in range(k + 1):
            factorial_i = self.factorial(i)
            cdf += (e ** -lambtha) * (lambtha ** i) / factorial_i
        return cdf

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
