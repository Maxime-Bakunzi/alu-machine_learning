#!/usr/bin/env python3
"""
The Module of converting one-hot matrix into a vector of labels
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): with shape (classes, m)

    Returns:
        A numpy.ndarray  with shape (m,) containing the numeric labels
        for each example, or None on failure.
    """

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # Use np.argmax to find the index of the maximum value in each column
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
