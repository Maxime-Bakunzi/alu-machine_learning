#!/usr/bin/env python3
"""
Bidirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell (BidirectionalCell): Instance of BidirectionalCell.
        X (numpy.ndarray): Data to be used, of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state in the forward direction, of shape (m, h).
        h_t (numpy.ndarray): Initial hidden state in the backward direction, of shape (m, h).

    Returns:
        H (numpy.ndarray): Concatenated hidden states, of shape (t, m, 2 * h).
        Y (numpy.ndarray): Outputs, of shape (t, m, o).
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = bi_cell.by.shape[1]

    # Initialize forward and backward hidden states
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward direction
    h_prev_forward = h_0
    for step in range(t):
        h_prev_forward = bi_cell.forward(h_prev_forward, X[step])
        H_forward[step] = h_prev_forward

    # Backward direction
    h_prev_backward = h_t
    for step in range(t - 1, -1, -1):
        h_prev_backward = bi_cell.backward(h_prev_backward, X[step])
        H_backward[step] = h_prev_backward

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward, H_backward), axis=-1)

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
