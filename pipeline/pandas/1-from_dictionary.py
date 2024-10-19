#!/usr/bin/env python3
"""
Module for creating a pandas Dataframe from a dictionary.
"""

import pandas as pd

# Create the dictionary
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create the DataFrame
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
