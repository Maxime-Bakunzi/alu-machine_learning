#!/usr/bin/env python3
"""
Module for Neural Style Transfer implementation
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs Neural Style Transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image):
        """
        Initialize the NST class
        """
        if not isinstance(style_image, np.ndarray):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = 1
        self.beta = 1

        self.load_model()

        # Calculate content feature
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        self.content_feature = self.model(preprocessed)[-1]

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim

        h, w = image.shape[:-1]
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.image.resize(image, (new_h, new_w), method='bicubic')
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(vgg.input, model_outputs)

    def gram_matrix(self, input_layer):
        """
        Calculate gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    def style_cost(self, style_outputs):
        """
        Calculate the style cost for generated image
        """
        if not isinstance(style_outputs, list):
            raise TypeError("style_outputs must be a list with a length of {}".format(
                len(self.style_layers)))

        if len(style_outputs) != len(self.style_layers):
            raise TypeError("style_outputs must be a list with a length of {}".format(
                len(self.style_layers)))

        style_targets = [self.gram_matrix(style_layer)
                         for style_layer in self.model(self.style_image * 255)[:-1]]

        weight = 1.0 / float(len(self.style_layers))
        style_cost = 0

        for target, output in zip(style_targets, style_outputs):
            gram_style = target
            gram_generated = self.gram_matrix(output)

            style_cost += weight * tf.reduce_mean(
                tf.square(gram_style - gram_generated))

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        Args:
            content_output: tf.Tensor containing the content output for
                          the generated image

        Returns:
            The content cost
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape))

        if content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape))

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing the
                           generated image

        Returns:
            (J, J_content, J_style)
            - J is the total cost
            - J_content is the content cost
            - J_style is the style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape))

        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape))

        # Preprocess generated image
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)

        # Get outputs for generated image
        outputs = self.model(preprocessed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # Calculate costs
        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        # Calculate total cost
        J = self.beta * J_style + self.alpha * J_content

        return J, J_content, J_style
