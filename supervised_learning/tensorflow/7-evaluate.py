#!/usr/bin/env python3
"""Module for evaluating the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluate teh output of aneural network

    Args:
        X (numpy.ndarray): The input data to evaluate.
        Y (numpy.ndarray): The one-hot labels for X.
        save_path (str): The location to load the model from.

    Returns:
        tuple: The network's prediction, acccuarcy, and loss, respectively.
    """
    with tf.Session() as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')

        # Restore the model
        saver.restore(sess, save_path)

        # Get the required tensors from collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # Evaluate the model
        prediction, acc, cost = sess.run([y_pred, accuracy, loss],
                                         feed_dict={x: X, y: Y})

        return prediction, acc, cost
