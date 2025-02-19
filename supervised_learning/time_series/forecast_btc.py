#!/usr/bin/env python3
"""
forecast_btc.py
Script that creates, trains, and validates a Keras model for forecasting 
BTC prices. The model uses the past 24 hours (1440 minutes) of BTC data 
to predict the BTC close price at the close of the following hour.
"""

import os
import numpy as np
import tensorflow as tf


def load_data(train_file, val_file):
    """
    Loads preprocessed training and validation data from npz files.

    Args:
        train_file (str): Path to the training npz file.
        val_file (str): Path to the validation npz file.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    with np.load(train_file) as data:
        X_train = data['X']
        y_train = data['y']
    with np.load(val_file) as data:
        X_val = data['X']
        y_val = data['y']
    return X_train, y_train, X_val, y_val


def create_dataset(X, y, batch_size, shuffle=True):
    """
    Creates a tf.data.Dataset from features and targets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: The dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.batch(batch_size)
    return dataset


def build_model(input_shape):
    """
    Builds and returns the RNN model.

    Args:
        input_shape (tuple): Shape of the input (timesteps, features).

    Returns:
        tf.keras.Model: The compiled model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    """
    Main function to load data, build the model, and train/validate.
    """
    # Paths to preprocessed data files
    train_file = 'train.npz'
    val_file = 'val.npz'

    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        print("Preprocessed data files not found. Please run "
              "preprocess_data.py first.")
        return

    X_train, y_train, X_val, y_val = load_data(train_file, val_file)

    # Create tf.data.Dataset objects
    batch_size = 32
    train_dataset = create_dataset(X_train, y_train, batch_size, shuffle=True)
    val_dataset = create_dataset(X_val, y_val, batch_size, shuffle=False)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Train model
    epochs = 10
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

    # Save the trained model
    model.save('forecast_btc_model.h5')
    print("Model saved as forecast_btc_model.h5")


if __name__ == '__main__':
    main()
