#!/usr/bin/env python3
"""Module for Neural Style Transfer with variational cost"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initialize the NST class
        Args:
            style_image: image used as style reference
            content_image: image used as content reference
            alpha: weight for content cost
            beta: weight for style cost
            var: weight for variational cost
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape "
                            "(h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape "
                            "(h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()

        self.gram_style_features = self.generate_features()[
            :len(self.style_layers)]
        self.content_feature = self.generate_features()[-1]

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = w * h_new // h
        else:
            w_new = 512
            h_new = h * w_new // w

        scaled = tf.image.resize_bicubic(image[None, ...],
                                         (h_new, w_new))[0]

        scaled = tf.clip_by_value(scaled / 255, 0, 1)

        return scaled

    def load_model(self):
        """Creates the model used to calculate cost"""
        vgg = tf.keras.applications.VGG19(include_top=False)
        x = vgg.input

        style_outputs = []
        for layer in self.style_layers:
            style_outputs.append(vgg.get_layer(layer).output)

        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        model = tf.keras.Model(x, outputs)
        model.trainable = False

        self.model = model

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        style_features = self.model(self.style_image)
        content_features = self.model(self.content_image)

        style_outputs = style_features[:-1]
        content_output = content_features[-1]

        gram_style_features = []
        for style_feature in style_outputs:
            gram_style_features.append(self.gram_matrix(style_feature))

        return gram_style_features + [content_output]

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor")

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError("generated_image must be a tensor")

        dx = generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]
        dy = generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]

        return tf.reduce_sum(tf.square(dx)) + tf.reduce_sum(tf.square(dy))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError("generated_image must be a tensor")

        style_outputs = self.model(generated_image)
        content_output = style_outputs[-1]
        style_outputs = style_outputs[:-1]

        content_cost = self.alpha * tf.reduce_mean(
            tf.square(content_output - self.content_feature))

        style_cost = 0
        weight = 1 / len(self.style_layers)
        for gram_style_feature, style_output in zip(
                self.gram_style_features, style_outputs):
            style_cost += weight * self.beta * tf.reduce_mean(
                tf.square(self.gram_matrix(style_output) - gram_style_feature))

        var_cost = self.var * self.variational_cost(generated_image)

        total_cost = content_cost + style_cost + var_cost

        return (total_cost, content_cost, style_cost, var_cost)

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image"""
        with tf.GradientTape() as tape:
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image)
        grad = tape.gradient(J_total, generated_image)
        return grad, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = self.content_image
        generated_image = tf.Variable(generated_image)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                           beta1=beta1,
                                           beta2=beta2)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grad, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)
            optimizer.apply_gradients([(grad, generated_image)])
            clipped = tf.clip_by_value(generated_image, 0, 1)
            generated_image.assign(clipped)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step and i % step == 0:
                print("Cost at iteration {}: {}, content {}, style {}, var {}"
                      .format(i, J_total, J_content, J_style, J_var))

        return best_image, best_cost
