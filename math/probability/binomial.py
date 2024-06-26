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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".

        Parameters:
        k (int): The number of "successes".

        Returns:
        float: The PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient
        bin_coeff = self.factorial(
                self.n)//(self.factorial(k)*self.factorial(self.n-k))

        # Calculate PMF
        pmf_value = bin_coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf_value

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".

        Parameters:
        k (int): The number of "successes".

        Returns:
        float: The CDF value for k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            k = self.n

        cdf_value = sum(self.pmf(i) for i in range(k + 1))
        return cdf_value

    def factorial(self, x):
        """
        Calculates the factorial of a number.

        Parameters:
        x (int): The number to calculate the factorial of.

        Returns:
        int: The factorial of x.
        """
        if x == 0 or x == 1:
            return 1
        factorial = 1
        for i in range(2, x + 1):
            factorial *= i
        return factorial
