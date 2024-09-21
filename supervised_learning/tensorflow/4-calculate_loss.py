#!/usr/bin/env python3
"""Module for calculating the softmax cross-entropy loss of a prediction"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculate the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): A placeholder for the labels of input data.
        y_pred (tf.Tensor): A tensor containing the network's predictions.

    Returns:
        tf.Tensors: A tensor containing the loss of the prediction.
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=y_pred))
    return loss
