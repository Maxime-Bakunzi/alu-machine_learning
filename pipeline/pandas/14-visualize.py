#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop('Weighted_Price', axis=1)

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp to date
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the dataframe on Date
df.set_index('Date', inplace=True)

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
for col in ['High', 'Low', 'Open']:
    df[col].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 and beyond, resample to daily intervals
df_daily = df['2017':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df_daily.index, df_daily['Open'], label='Open')
ax.plot(df_daily.index, df_daily['High'], label='High')
ax.plot(df_daily.index, df_daily['Low'], label='Low')
ax.plot(df_daily.index, df_daily['Close'], label='Close')
ax.plot(df_daily.index, df_daily['Volume_(BTC)'], label='Volume_(BTC)')
ax.plot(df_daily.index,
        df_daily['Volume_(Currency)'], label='Volume_(Currency)')

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Bitcoin Data from 2017 to 2019')
ax.legend()

plt.tight_layout()
plt.show()
