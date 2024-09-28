#!/usr/bin/env python3
"""Module for creating batch normalization layer in TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function to be used on the output
     of the layer.

    Returns:
    tf.Tensor: The tensor of the activated output for the layer.
    """
    # Define the Dense layer with the variance scaling initializer
    base_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"))

    # Create the dense layer
    Z = base_layer(prev)

    # Calculate the mean and variance of the layer
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Initialize gamma and beta
    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")

    # Apply batch normalization
    epsilon = 1e-8
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)

    # Apply activation function
    return activation(Z_norm)
