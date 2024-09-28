#!/usr/bin/env python3
"""
Module containing a complete neural network model with various optimizations
"""

import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow using Adam optimization, 
    mini-batch gradient descent, learning rate decay, and batch normalization.

    Data_train: tuple (X_train, Y_train), training inputs and labels
    Data_valid: tuple (X_valid, Y_valid), validation inputs and labels
    layers: list of number of nodes for each layer
    activations: list of activation functions for each layer
    alpha: learning rate
    beta1: first moment weight for Adam optimization
    beta2: second moment weight for Adam optimization
    epsilon: small number to avoid division by zero
    decay_rate: decay rate for inverse time decay of the learning rate
    batch_size: size of mini-batches for training
    epochs: number of epochs to train the model
    save_path: path to save the trained model
    Returns: the path where the model was saved
    """

    # Unpack the training and validation data
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    num_features = X_train.shape[1]
    num_classes = Y_train.shape[1]

    # Placeholders for inputs and labels
    x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")

    # Initialize the global step (used for learning rate decay)
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # Apply learning rate decay
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_rate, 1)

    # Build the network with layers and activations
    output = x
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        output = tf.layers.Dense(units=nodes,
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                     mode="FAN_AVG"),
                                 name=f"layer_{i}")(output)
        # Apply batch normalization
        gamma = tf.Variable(tf.ones([nodes]), name=f"gamma_{
                            i}", trainable=True)
        beta = tf.Variable(tf.zeros([nodes]), name=f"beta_{i}", trainable=True)
        mean, variance = tf.nn.moments(output, axes=[0])
        output = tf.nn.batch_normalization(
            output, mean, variance, beta, gamma, variance_epsilon=1e-8)

        # Apply the activation function
        if activation is not None:
            output = activation(output)

    # Compute the loss using softmax cross entropy
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

    # Define the Adam optimizer with the given parameters
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Define metrics for accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing all variables
    init = tf.global_variables_initializer()

    # Saver to save the model
    saver = tf.train.Saver()

    # Start a TensorFlow session to run the model
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for epoch in range(epochs):
            # Shuffle the training data before each epoch
            shuffle_idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffle_idx]
            Y_train = Y_train[shuffle_idx]

            print(f"After {epoch} epochs:")
            # Run over mini-batches
            for step in range(0, X_train.shape[0], batch_size):
                # Determine the batch size
                end = step + batch_size
                if end > X_train.shape[0]:
                    end = X_train.shape[0]

                X_batch = X_train[step:end]
                Y_batch = Y_train[step:end]

                # Train the mini-batch
                _, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy],
                                                         feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {batch_loss}")
                    print(f"\t\tAccuracy: {batch_accuracy}")

            # After each epoch, print the training and validation costs/accuracies
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the trained model
        save_path = saver.save(sess, save_path)

    return save_path
