#!/usr/bin/env python3
"""
Normal distribution class module
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the normal distribution

        Args:
            data: list of the data to be used to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            ValueError: If stddev is not positive
             or data has less than two values
            TypeError: If data is not a list
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x-self.mean)**2 for x in data)/len(data))**0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x (float): The x-value

        Returns:
            float: The z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z (float): The z-score

        Returns:
            float: The x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x (float): The x-value

        Returns:
            float: The PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        exp = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return coef * exp

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x (float): The x-value

        Returns:
            float: The CDF value for x
        """
        # Constants for the approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        # Save the sign of x
        sign = 1 if x >= self.mean else -1
        x = abs(x - self.mean) / (self.stddev * (2 ** 0.5))

        # A&S formula 7.1.26 approximation for erf
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*(2.7182818285**(-x * x))

        return 0.5 * (1.0 + sign * y)
