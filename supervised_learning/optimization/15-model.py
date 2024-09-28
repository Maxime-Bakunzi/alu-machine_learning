#!/usr/bin/env python3
"""
Module containing a complete neural network model with various optimizations
"""

import numpy as np
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer for the neural network"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for the neural network"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer,
                            name='layer', activation=None)
    x = layer(prev)

    if activation is None:
        return x

    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)

    return activation(bn)


def forward_prop(x, layers, activations):
    """Forward propagation with batch normalization"""
    for i in range(len(layers)):
        if i != len(layers) - 1:
            x = create_batch_norm_layer(x, layers[i], activations[i])
        else:
            x = create_layer(x, layers[i], activations[i])
    return x


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of the prediction"""
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    y_pred = forward_prop(x, layers, activations)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate)

    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: Data_train[0], y: Data_train[1]}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: Data_valid[0], y: Data_valid[1]}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffle, Y_shuffle = shuffle_data(
                    Data_train[0], Data_train[1])

                for step in range(0, X_shuffle.shape[0], batch_size):
                    X_batch = X_shuffle[step:step+batch_size]
                    Y_batch = Y_shuffle[step:step+batch_size]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if (step // batch_size + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(step//batch_size + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
