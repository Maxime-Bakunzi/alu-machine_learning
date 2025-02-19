#!/usr/bin/env python3
"""
Bidirectional Output
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell.
    """

    def __init__(self, i, h, o):
        """
        Initializes the bidirectional RNN cell.

        Args:
            i (int): Number of input features.
            h (int): Number of hidden units.
            o (int): Number of output units.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Args:
            H (numpy.ndarray): Concatenated hidden states from both directions,
                              of shape (t, m, 2 * h).

        Returns:
            numpy.ndarray: The outputs of shape (t, m, o).
        """
        t, m, _ = H.shape
        o = self.by.shape[1]

        # Initialize the outputs array
        Y = np.zeros((t, m, o))

        for step in range(t):
            # Compute the output for each time step
            Y[step] = np.dot(H[step], self.Wy) + self.by

        return Y
