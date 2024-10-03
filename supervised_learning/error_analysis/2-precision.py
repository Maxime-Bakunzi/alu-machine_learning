#!/usr/bin/env python3
"""Module for calculating precision from a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion nump.array of shape
                                    (classes, classes) where row indices
                                    represent the correct labels and column
                                    indices represent the predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the precision
                        of each classes.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives
