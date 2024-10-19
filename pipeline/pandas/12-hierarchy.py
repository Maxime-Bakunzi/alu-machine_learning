#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Set Timestamp as index for both DataFrames
df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

# Filter data between timestamps 1417411980 and 1417417980, inclusive
df1_filtered = df1.loc[1417411980:1417417980]
df2_filtered = df2.loc[1417411980:1417417980]

# Concatenate the filtered DataFrames
df = pd.concat([df2_filtered, df1_filtered], keys=[
               'bitstamp', 'coinbase'], axis=0)

# Rearrange the MultiIndex levels
df = df.swaplevel(0, 1)

# Sort the index to display rows in chronological order
df.sort_index(inplace=True)

print(df)
