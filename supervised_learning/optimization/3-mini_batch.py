#!/usr/bin/env python3
"""Module for training a neural network model using mini-batch
gradient descent"""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
    X_train (numpy.ndarray): Training data of shape (m, 784)
    Y_train (numpy.ndarray): Training labels of shape (m, 10)
    X_valid (numpy.ndarray): Validation data of shape (m, 784)
    Y_valid (numpy.ndarray): Validation labels of shape (m, 10)
    batch_size (int): Number of data points in a batch
    epochs (int): Number of times to train on the entire dataset
    load_path (str): Path from which to load the model
    save_path (str): Path to where the model should be saved after training

    Returns:
    str: The path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]
        steps = m // batch_size + (m % batch_size != 0)

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for step in range(steps):
                    start = step * batch_size
                    end = min(start + batch_size, m)
                    X_batch = X_shuffled[start:end]
                    Y_batch = Y_shuffled[start:end]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if (step + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
