#!/usr/bin/env python3
"""
Module containing a complete neural network model with various optimizations
"""

import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for the tensorflow session
    
    Args:
        nx (int): number of feature columns in our data
        classes (int): number of classes in our classifier

    Returns:
        (tf.placeholder, tf.placeholder): placeholders for the input data and labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y


def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network
    
    Args:
        prev: tensor output of the previous layer
        n (int): number of nodes in the layer to create
        activation: activation function to use

    Returns:
        tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)


def forward_prop(x, layers, activations):
    """
    Creates the forward propagation graph for the neural network
    
    Args:
        x: placeholder for the input data
        layers (list): list containing the number of nodes in each layer
        activations (list): list containing the activation functions for each layer

    Returns:
        tensor output of the last layer in the neural network
    """
    for i in range(len(layers)):
        if i != len(layers) - 1:
            x = create_layer(x, layers[i], activations[i])
        else:
            x = create_layer(x, layers[i], None)
    return x


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the prediction
    
    Args:
        y: placeholder for the labels
        y_pred: tensor containing the network's predictions

    Returns:
        tensor containing the decimal accuracy of the prediction
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of the prediction
    
    Args:
        y: placeholder for the labels
        y_pred: tensor containing the network's predictions

    Returns:
        tensor containing the loss of the prediction
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    
    Args:
        X: numpy.ndarray of shape (m, nx) containing the input data
        Y: numpy.ndarray of shape (m, classes) containing the labels

    Returns:
        the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization
    
    Args:
        Data_train (tuple): contains the training inputs and labels
        Data_valid (tuple): contains the validation inputs and labels
        layers (list): contains the number of nodes in each layer
        activations (list): contains the activation functions for each layer
        alpha (float): learning rate
        beta1 (float): weight for the first moment in Adam optimization
        beta2 (float): weight for the second moment in Adam optimization
        epsilon (float): small number to avoid division by zero
        decay_rate (int): decay rate for inverse time decay of the learning rate
        batch_size (int): number of data points in the mini-batch
        epochs (int): number of times the training should pass through the whole dataset
        save_path (str): path where the model should be saved

    Returns:
        str: the path where the model was saved
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
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
