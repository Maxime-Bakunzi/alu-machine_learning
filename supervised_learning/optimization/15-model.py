#!/usr/bin/env python3
"""Module for building, training, and saving a neural network model in
 TensorFlow"""

import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (callable): The activation function to be applied.

    Returns:
    tf.Tensor: The activated output of the batch normalization layer.
    """
    # Initialize the dense layer
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer)

    # Compute the output of the layer
    Z = layer(prev)

    # Calculate batch normalization
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon)

    # Apply the activation function
    if activation is not None:
        return activation(batch_norm)
    return batch_norm


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
     Adam optimization,
    mini-batch gradient descent, learning rate decay, and batch normalization.

    Args:
    Data_train (tuple): Training data containing inputs and labels.
    Data_valid (tuple): Validation data containing inputs and labels.
    layers (list): Number of nodes in each layer.
    activations (list): Activation functions for each layer.
    alpha (float): Learning rate.
    beta1 (float): Weight for the first moment of Adam optimizer.
    beta2 (float): Weight for the second moment of Adam optimizer.
    epsilon (float): Small number to prevent division by zero.
    decay_rate (float): Decay rate for learning rate decay.
    batch_size (int): Number of data points per mini-batch.
    epochs (int): Number of training iterations over the entire dataset.
    save_path (str): Path to save the trained model.

    Returns:
    str: Path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    m = X_train.shape[0]
    n_classes = Y_train.shape[1]

    # Create placeholders
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name="x")
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name="y")

    # Create learning rate with decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, decay_rate, staircase=True)

    # Build the neural network layers
    prev_layer = x
    for i, layer_size in enumerate(layers):
        prev_layer = create_batch_norm_layer(
            prev_layer, layer_size, activations[i])

    # Loss, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prev_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(prev_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training loop
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(m)
            X_shuffled, Y_shuffled = X_train[shuffle_idx], Y_train[shuffle_idx]

            for step in range(0, m, batch_size):
                end = step + batch_size if step + batch_size <= m else m
                X_batch, Y_batch = X_shuffled[step:end], Y_shuffled[step:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0 and step > 0:
                    step_loss, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_loss))
                    print("\t\tAccuracy: {}".format(step_accuracy))

            # After each epoch, print training and validation accuracy and loss
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
