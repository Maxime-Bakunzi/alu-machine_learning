#!/usr/bin/env python3
"""
Module containing the implementation of learning rate decay using inverse
 time decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Args:
        alpha: The original learning rate
        decay_rate: The weight used to determine the rate at which alpha
         will decay
        global_step: The number of passes of gradient descent that have elapsed
        decay_step: The number of passes of gradient descent that should occur
                    before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # Calculate the number of times the learning rate should have been decayed
    decay_times = np.floor(global_step / decay_step)

    # Calculate the new learning rate using inverse time decay formula
    new_alpha = alpha / (1 + decay_rate * decay_times)

    return new_alpha
