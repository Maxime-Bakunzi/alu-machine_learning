#!/usr/bin/env python3
"""
Module containing the implementation of learning rate decay using inverse
 time decay in TensorFlow
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
     decay.

    Args:
        alpha: The original learning rate
        decay_rate: The weight used to determine the rate at which alpha
         will decay
        global_step: The number of passes of gradient descent that have
         elapsed
        decay_step: The number of passes of gradient descent that shoul
         occur before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
