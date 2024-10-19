#!/usr/bin/env python3
"""
Script to fill missing data points in a pandas DataFrame
according to specific rules.
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Fill missing values in Close with the previous row value
df['Close'] = df['Close'].ffill()

# Fill missing values in High, Low, Open with the same row's Close value
for column in ['High', 'Low', 'Open']:
    df[column] = df[column].fillna(df['Close'])

# Fill missing values in Volume_(BTC) and Volume_(Currency) with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

print(df.head())
print(df.tail())
