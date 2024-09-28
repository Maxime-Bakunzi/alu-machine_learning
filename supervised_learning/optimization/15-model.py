#!/usr/bin/env python3
"""
Module containing a complete neural network model with various optimizations
"""

import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Create placeholders
    X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    Y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]])

    # Create variables
    weights = []
    biases = []
    for i in range(len(layers)):
        if i == 0:
            w = tf.Variable(tf.random_normal([X_train.shape[1], layers[i]]))
        else:
            w = tf.Variable(tf.random_normal([layers[i-1], layers[i]]))
        b = tf.Variable(tf.zeros([layers[i]]))
        weights.append(w)
        biases.append(b)

    # Build the network
    layer = X
    for i in range(len(layers)):
        z = tf.matmul(layer, weights[i]) + biases[i]
        if i < len(layers) - 1:
            z = tf.layers.batch_normalization(z, training=True)
            layer = activations[i](z)
        else:
            layer = z

    # Define loss and optimizer
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=Y))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_steps=1, decay_rate=decay_rate)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle training data
            shuffle_indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[shuffle_indices]
            Y_shuffled = Y_train[shuffle_indices]

            # Mini-batch training
            for step in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[step:step+batch_size]
                Y_batch = Y_shuffled[step:step+batch_size]

                _, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy],
                                                         feed_dict={X: X_batch, Y: Y_batch})

                if (step // batch_size + 1) % 100 == 0:
                    print("\tStep {}:".format(step // batch_size + 1))
                    print("\t\tCost: {}".format(batch_loss))
                    print("\t\tAccuracy: {}".format(batch_accuracy))

            # Print epoch results
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={X: X_train, Y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={X: X_valid, Y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
