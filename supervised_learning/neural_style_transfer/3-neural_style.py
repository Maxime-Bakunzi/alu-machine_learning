#!/usr/bin/env python3
"""
Neural Style Transfer Implementation
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class
        Args:
            style_image: image used as style reference (numpy.ndarray)
            content_image: image used as content reference (numpy.ndarray)
            alpha: weight for content cost
            beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')

        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

        # Generate and set the features
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image
        Returns:
            scaled image as a tf.tensor
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape

        # Determine new dimensions while maintaining aspect ratio
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        # Resize image using bicubic interpolation
        image = tf.image.resize_bicubic(
            tf.expand_dims(image, 0), (h_new, w_new))

        # Scale pixel values to range [0, 1]
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as base
        Returns:
            model: the keras model used to calculate cost
        """
        # Load the VGG19 model without top layers
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        # Get the outputs for style and content layers
        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(self.content_layer).output]
        outputs = style_outputs + content_outputs

        # Create the model
        model = tf.keras.Model(vgg.input, outputs)

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates gram matrix of a layer
        Args:
            input_layer: instance of tf.Tensor or tf.Variable of shape
                        (1, h, w, c) containing the layer output
        Returns:
            tf.Tensor of shape (1, c, c) containing the gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get the number of channels from the input shape
        channels = int(input_layer.shape[-1])

        # Reshape the layer to 2D matrix where each row represents a position (h,w)
        # and each column represents a channel
        a = tf.reshape(input_layer, [-1, channels])

        # Calculate gram matrix
        n = tf.cast(tf.shape(a)[0], tf.float32)
        gram = tf.matmul(a, a, transpose_a=True) / n

        # Add batch dimension
        gram = tf.expand_dims(gram, axis=0)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets:
            gram_style_features: list of gram matrices from style layer outputs
            content_feature: content layer output of the content image
        """
        # Preprocess images using VGG19 preprocessing
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Get style and content features
        outputs = self.model(style_image)
        style_outputs = outputs[:-1]

        # Calculate gram matrices for style features
        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]

        # Get content feature
        outputs = self.model(content_image)
        self.content_feature = outputs[-1]
