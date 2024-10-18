#!/usr/bin/env python3
"""Module for creating a pandas DataFrame from a Numpy array."""

import pandas as pd
import string


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Args:
        array (np.ndarray): The Numpy array to convert.

    Returns:
        pd.DataFrame: The newly created pandas DataFrame.
    """
    num_columns = array.shape[1]
    columns = list(string.ascii_uppercase[:num_columns])
    return pd.DataFrame(array, columns=columns)
