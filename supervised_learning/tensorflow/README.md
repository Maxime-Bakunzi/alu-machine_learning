# TensorFlow Neural Network Project

This project implements a neural network classifier using TensorFlow. It includes various components for building, training, and evaluating a neural network model.

## Project Structure

The project consists of the following Python scripts:

1. `0-create_placeholders.py`: Creates placeholders for the neural network.
2. `1-create_layer.py`: Creates a layer for the neural network.
3. `2-forward_prop.py`: Implements forward propagation for the neural network.
4. `3-calculate_accuracy.py`: Calculates the accuracy of the neural network's predictions.
5. `4-calculate_loss.py`: Calculates the softmax cross-entropy loss of the predictions.
6. `5-create_train_op.py`: Creates the training operation for the neural network.
7. `6-train.py`: Builds, trains, and saves the neural network classifier.
8. `7-evaluate.py`: Evaluates the output of the trained neural network.

## Requirements

- Python 3.5
- NumPy 1.15
- TensorFlow 1.12

## Usage

Each script can be run independently for testing purposes. For example:

```bash
./0-main.py
./1-main.py
# ... and so on
```

To train the model:

```bash
./6-main.py
```

To evaluate the model:

```bash
./7-main.py
```

## Task Descriptions

### 0. Placeholders

Creates placeholders for the neural network input data and labels.

### 1. Layers

Creates a layer for the neural network with a specified number of nodes and activation function.

### 2. Forward Propagation

Implements forward propagation for the neural network.

### 3. Accuracy

Calculates the accuracy of the neural network's predictions.

### 4. Loss

Calculates the softmax cross-entropy loss of the predictions.

### 5. Train Op

Creates the training operation for the network using gradient descent.

### 6. Train

Builds, trains, and saves the neural network classifier. It prints the training progress and saves the model.

### 7. Evaluate

Evaluates the output of the trained neural network on new data.

## Note

This project is designed to work with TensorFlow 1.12. Make sure you have the correct version installed before running the scripts.
