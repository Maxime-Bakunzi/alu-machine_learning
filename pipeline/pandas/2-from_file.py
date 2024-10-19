#!/usr/bin/env python3
"""
Module for loading data from a file as a pandas DataFrame.
"""

import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.Dataframe.

    Args:
        filename (str): The file to load from.
        delimiter (str): The column separator.

    Returns:
        pd.Dataframe: The loaded Pandas DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
