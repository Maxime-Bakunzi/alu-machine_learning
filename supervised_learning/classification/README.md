# ALU Machine Learning Projects: Classification

This repository contains tasks from the ALU Machine Learning curriculum, focusing on building and training neural networks from scratch. The projects start from basic concepts like implementing a single neuron to building deep neural networks capable of multiclass classification with various activation functions.

## Table of Contents

1. [0. Neuron](#0-neuron)
2. [1. Privatize Neuron](#1-privatize-neuron)
3. [2. Neuron Forward Propagation](#2-neuron-forward-propagation)
4. [3. Neuron Cost](#3-neuron-cost)
5. [4. Evaluate Neuron](#4-evaluate-neuron)
6. [5. Neuron Gradient Descent](#5-neuron-gradient-descent)
7. [6. Train Neuron](#6-train-neuron)
8. [7. Upgrade Train Neuron](#7-upgrade-train-neuron)
9. [8. Deep Neural Network](#8-deep-neural-network)
10. [9. Privatize Deep Neural Network](#9-privatize-deep-neural-network)
11. [10. Deep Neural Network Forward Propagation](#10-deep-neural-network-forward-propagation)
12. [11. Deep Neural Network Cost](#11-deep-neural-network-cost)
13. [12. Evaluate Deep Neural Network](#12-evaluate-deep-neural-network)
14. [13. Deep Neural Network Gradient Descent](#13-deep-neural-network-gradient-descent)
15. [14. Train Deep Neural Network](#14-train-deep-neural-network)
16. [15. Save and Load Neural Network](#15-save-and-load-neural-network)
17. [16. One-Hot Encode](#16-one-hot-encode)
18. [17. One-Hot Decode](#17-one-hot-decode)
19. [18. Deep Neural Network with Multiclass Classification](#18-deep-neural-network-with-multiclass-classification)
20. [19. Learning Rate Decay](#19-learning-rate-decay)
21. [20. Batch Normalization](#20-batch-normalization)
22. [21. Dropout Regularization](#21-dropout-regularization)
23. [22. Build a Deep Neural Network from Scratch](#22-build-a-deep-neural-network-from-scratch)
24. [23. Train Deep Neural Network with Data Augmentation](#23-train-deep-neural-network-with-data-augmentation)
25. [24. One-Hot Encoding](#24-one-hot-encoding)
26. [25. One-Hot Decoding](#25-one-hot-decoding)
27. [26. Persistent Neural Network Models](#26-persistent-neural-network-models)
28. [27. Multiclass Classification with Deep Neural Networks](#27-multiclass-classification-with-deep-neural-networks)
29. [28. Different Activation Functions](#28-different-activation-functions)

---

## 0. Neuron

This task implements a single neuron for binary classification. The neuron includes methods for forward propagation, cost calculation, and gradient descent.

- **Files**: `0-neuron.py`
- **Key methods**: `__init__`, `forward_prop`, `cost`, `evaluate`, `gradient_descent`

## 1. Privatize Neuron

We update the `Neuron` class to privatize its attributes. This is a key step in ensuring that internal properties are not directly accessed, but through getter methods.

- **Files**: `1-neuron.py`
- **Key methods**: `__init__`, `__private_attributes__`

## 2. Neuron Forward Propagation

Here we implement forward propagation for the neuron. This involves computing the linear combination of inputs and applying the sigmoid activation function.

- **Files**: `2-neuron.py`
- **Key methods**: `forward_prop`

## 3. Neuron Cost

In this task, we implement a cost function using logistic regression to measure the performance of the model during training.

- **Files**: `3-neuron.py`
- **Key methods**: `cost`

## 4. Evaluate Neuron

We extend the `Neuron` class to include an evaluation method that returns the predicted label and the cost function.

- **Files**: `4-neuron.py`
- **Key methods**: `evaluate`

## 5. Neuron Gradient Descent

The gradient descent algorithm is implemented in this task to update the weights of the neuron based on the computed gradients.

- **Files**: `5-neuron.py`
- **Key methods**: `gradient_descent`

## 6. Train Neuron

In this task, we implement the `train` method, which repeatedly applies forward propagation and gradient descent over a number of iterations.

- **Files**: `6-neuron.py`
- **Key methods**: `train`

## 7. Upgrade Train Neuron

We enhance the training process by including features like early stopping or validation data.

- **Files**: `7-neuron.py`
- **Key methods**: `train`

## 8. Deep Neural Network

We extend the neuron model to handle multiple layers in this task, building a deep neural network from scratch.

- **Files**: `8-deep_neural_network.py`
- **Key methods**: `__init__`, `forward_prop`, `cost`

## 9. Privatize Deep Neural Network

This task privatizes the attributes of the `DeepNeuralNetwork` class, similar to what was done for the single neuron in task 1.

- **Files**: `9-deep_neural_network.py`

## 10. Deep Neural Network Forward Propagation

We implement forward propagation for a deep neural network, applying the sigmoid activation function at each layer.

- **Files**: `10-deep_neural_network.py`
- **Key methods**: `forward_prop`

## 11. Deep Neural Network Cost

The cost function for the deep neural network is implemented here, following a similar method as in task 3 but extended for multiple layers.

- **Files**: `11-deep_neural_network.py`
- **Key methods**: `cost`

## 12. Evaluate Deep Neural Network

We extend the deep neural network class to include an evaluation function that outputs predictions and the cost.

- **Files**: `12-deep_neural_network.py`
- **Key methods**: `evaluate`

## 13. Deep Neural Network Gradient Descent

Gradient descent for deep neural networks is implemented in this task, allowing for weight updates across multiple layers.

- **Files**: `13-deep_neural_network.py`
- **Key methods**: `gradient_descent`

## 14. Train Deep Neural Network

The deep neural network training method is implemented, combining forward propagation, cost calculation, and gradient descent over multiple iterations.

- **Files**: `14-deep_neural_network.py`
- **Key methods**: `train`

## 15. Save and Load Neural Network

This task introduces methods to save and load deep neural networks to and from files.

- **Files**: `15-deep_neural_network.py`
- **Key methods**: `save`, `load`

## 16. One-Hot Encode

We implement one-hot encoding, converting labels into binary matrix representations that can be used for classification.

- **Files**: `16-one_hot_encode.py`
- **Key methods**: `one_hot_encode`

## 17. One-Hot Decode

The reverse of one-hot encoding is implemented here, converting a one-hot encoded matrix back to its original labels.

- **Files**: `17-one_hot_decode.py`
- **Key methods**: `one_hot_decode`

## 18. Deep Neural Network with Multiclass Classification

This task extends the deep neural network to handle multiclass classification using one-hot encoded outputs.

- **Files**: `18-deep_neural_network.py`

## 19. Learning Rate Decay

We implement learning rate decay to improve the training performance of deep neural networks by reducing the learning rate over time.

- **Files**: `19-deep_neural_network.py`
- **Key methods**: `train`

## 20. Batch Normalization

Batch normalization is added to the deep neural network to normalize inputs to each layer, improving stability and performance.

- **Files**: `20-deep_neural_network.py`

## 21. Dropout Regularization

Dropout regularization is implemented to prevent overfitting in the deep neural network by randomly deactivating neurons during training.

- **Files**: `21-deep_neural_network.py`

## 22. Build a Deep Neural Network from Scratch

We build a deep neural network from scratch, applying all the learned techniques from previous tasks.

- **Files**: `22-deep_neural_network.py`

## 23. Train Deep Neural Network with Data Augmentation

Data augmentation is added to the training process to improve the generalization of the model by artificially expanding the dataset.

- **Files**: `23-deep_neural_network.py`

## 24. One-Hot Encoding

This is an extended task focused on handling one-hot encoding for larger datasets.

- **Files**: `24-one_hot_encode.py`

## 25. One-Hot Decoding

We decode one-hot encoded vectors back to the original labels in this task.

- **Files**: `25-one_hot_decode.py`

## 26. Persistent Neural Network Models

This task builds on task 15 by making neural network models persistent, allowing for saving and reloading models without retraining.

- **Files**: `26-deep_neural_network.py`

## 27. Multiclass Classification with Deep Neural Networks

We expand on multiclass classification by further refining the deep neural network architecture.

- **Files**: `27-deep_neural_network.py`

## 28. Different Activation Functions

This task introduces alternative activation functions (`tanh`, `sigmoid`) for the hidden layers of the deep neural network.

- **Files**: `28-deep_neural_network.py`

---

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Maxime-Bakunzi/alu-machine_learning.git
    cd alu-machine_learning
    ```

2. **Install required packages**:
    ```bash
    pip install numpy matplotlib
    ```

3. **Run any of the scripts**:
    ```bash
    python3 28-main.py
    ```

---


