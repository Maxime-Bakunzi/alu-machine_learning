#!/usr/bin/env python3
"""Module for calculating the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate the accuracy of a prediction.

    Args:
        y (tf.Tensor): A placeholder for the labels of the input data.
        y_prep (tf.Tensor): A tensor containing the network's predictions.

    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the prection.
    """

    # Get the index of the highest value (the predicted class)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate the mean of correct predictions (accuracy)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
