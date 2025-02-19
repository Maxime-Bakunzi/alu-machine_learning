#!/usr/bin/env python3
"""
Bidirectional Cell Backward
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

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step.

        Args:
            h_next (numpy.ndarray): The next hidden state of shape (m, h).
            x_t (numpy.ndarray): The data input for the cell of shape (m, i).

        Returns:
            numpy.ndarray: The previous hidden state of shape (m, h).
        """
        m, i = x_t.shape
        h = h_next.shape[1]

        # Concatenate x_t and h_next
        concat = np.concatenate((x_t, h_next), axis=1)

        # Compute the previous hidden state
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev
