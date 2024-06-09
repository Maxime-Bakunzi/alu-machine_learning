#!/usr/bin/env python3
"""
Exponential distribution class module
"""


class Exponential:
    """
    Represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the exponential distribution

        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurrences in a given time frame

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
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x (float): The time period

        Returns:
            float: The PDF value for x
        """
        if x < 0:
            return 0

        # Using the formula f(x; lambtha) = lambtha * e^(-lambtha * x)
        e = 2.7182818285
        lambtha = self.lambtha

        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Args:
            x (float): The time period

        Returns:
            float: The CDF value for x
        """
        if x < 0:
            return 0

        # Using the formula F(x; lambtha) = 1 - e^(-lambtha * x)
        e = 2.7182818285
        lambtha = self.lambtha

        cdf = 1 - (e ** (-lambtha * x))
        return cdf
