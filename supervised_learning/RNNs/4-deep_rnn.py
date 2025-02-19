#!/usr/bin/env python3
"""
Deep RNN Module
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances of length l.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h).

    Returns:
        H (numpy.ndarray): All hidden states of shape (t + 1, l, m, h).
        Y (numpy.ndarray): All outputs of shape (t, m, o).
    """
    t, m, i = X.shape
    l = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]

    # Initialize arrays to store hidden states and outputs
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0

    # Perform forward propagation for each time step
    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            H[step + 1, layer] = h_next
            x_t = h_next  # Input for the next layer is the hidden state of the current layer
        Y[step] = y  # Output of the last layer

    return H, Y
