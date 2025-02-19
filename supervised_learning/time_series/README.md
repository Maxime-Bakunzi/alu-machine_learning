# BTC Price Forecasting

This project implements a time series forecasting model to predict the Bitcoin 
(BTC) price at the close of the hour following a 24-hour period of data. The 
project uses a Recurrent Neural Network (RNN) built with TensorFlow and Keras, 
and preprocessed data from Coinbase and Bitstamp.

## Project Structure

```
alu-machine_learning/
└── supervised_learning/
    └── time_series/
        ├── forecast_btc.py
        ├── preprocess_data.py
        └── README.md
```

## Requirements

- Ubuntu 16.04 LTS  
- Python 3.5  
- NumPy 1.15  
- TensorFlow 1.12  
- pycodestyle 2.4

**Note:** Use `print("{}".format(var))` for printing to ensure Python 3.5 
compatibility.

## Files Description

- **preprocess_data.py**: Combines, normalizes, and creates sliding windows 
  from raw BTC data CSV files. The processed data is split into training and 
  validation sets and saved as `train.npz` and `val.npz`.
  
- **forecast_btc.py**: Loads the preprocessed data, builds an RNN model using an 
  LSTM layer, and trains the model using a `tf.data.Dataset` pipeline. The 
  trained model is saved as `forecast_btc_model.h5`.

## Usage

1. **Preprocess Data**

   Run the following command with your raw CSV files (for example, 
   `coinbase.csv` and `bitstamp.csv`):

   ```
   ./preprocess_data.py coinbase.csv bitstamp.csv
   ```

   This will generate `train.npz` and `val.npz` in the current directory.

2. **Train and Validate the Model**

   After preprocessing, run:

   ```
   ./forecast_btc.py
   ```

   The script will load the preprocessed data, train the model, and save the 
   trained model as `forecast_btc_model.h5`.
