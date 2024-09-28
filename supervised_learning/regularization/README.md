# Regularization Techniques in Neural Networks

This project demonstrates various **regularization techniques** that help prevent overfitting in neural networks, improve generalization, and ensure models perform well on unseen data. The project covers L2 regularization, L1 regularization, dropout, and early stopping, with implementations in TensorFlow.

## Project Structure

```bash
.
├── 0-l2_reg_cost.py
├── 1-l2_reg_gradient_descent.py
├── 2-l2_reg_cost.py
├── 3-l2_reg_create_layer.py
├── 4-dropout_forward_prop.py
├── 5-dropout_gradient_descent.py
├── 6-dropout_create_layer.py
├── 7-early_stopping.py
└── README.md
```

## Files and Explanations

### 0. `0-l2_reg_cost.py`
This file contains a function that calculates the **L2 Regularization cost** of a neural network. L2 regularization penalizes large weights by adding the sum of squared weights to the cost function. This helps the network avoid overfitting by discouraging complexity.

- **Function**: `def l2_reg_cost(cost, lambtha, weights, L, m):`
- **Parameters**:
  - `cost`: baseline cost of the network without regularization.
  - `lambtha`: regularization parameter (lambda).
  - `weights`: dictionary containing the weights of the neural network.
  - `L`: number of layers in the neural network.
  - `m`: number of training examples.
- **Returns**: the cost of the network accounting for L2 regularization.

### 1. `1-l2_reg_gradient_descent.py`
This file implements **L2 Regularization** in **Gradient Descent**. During gradient descent, the weights are updated not just based on the gradient but also with a penalty for larger weights, which is proportional to lambda.

- **Function**: `def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):`
- **Parameters**:
  - `Y`: correct labels for the training data.
  - `weights`: dictionary containing the weights of the neural network.
  - `cache`: dictionary of outputs of each layer.
  - `alpha`: learning rate.
  - `lambtha`: regularization parameter.
  - `L`: number of layers in the network.
- **Returns**: The function updates the weights of the neural network with L2 regularization.

### 2. `2-l2_reg_cost.py`
This script calculates the cost of a neural network with L2 regularization

- **Function**: `def l2_reg_cost(cost):`
- **Parameters**:
  - `cost`: The cost of the network without L2 regularization.
- **Returns**: The cost of the network accounting for L2 regularization.

### 3. `3-l2_reg_create_layer.py`
This script creates a **TensorFlow layer** that includes **L2 Regularization** using the `tf.contrib.layers.l2_regularizer`.

- **Function**: `def l2_reg_create_layer(prev, n, activation, lambtha):`
- **Parameters**:
  - `prev`: tensor output of the previous layer.
  - `n`: number of nodes in the layer.
  - `activation`: activation function to be applied (e.g., ReLU).
  - `lambtha`: regularization parameter.
- **Returns**: the output of the new layer with L2 regularization.

### 4. `4-dropout_forward_prop.py`
This script performs **forward propagation** in a neural network using **dropout**. Dropout randomly "drops" neurons during training, which forces the network to learn redundant representations, thus reducing overfitting.

- **Function**: `def dropout_forward_prop(X, weights, L, keep_prob):`
- **Parameters**:
  - `X`: input data.
  - `weights`: dictionary containing the weights of the neural network.
  - `L`: number of layers in the network.
  - `keep_prob`: probability of keeping a neuron active during dropout.
- **Returns**: the output of the forward propagation with dropout.

### 5. `5-dropout_gradient_descent.py`
This file implements **Gradient Descent** with **Dropout**. After applying forward propagation with dropout, the weights are updated while considering which neurons were kept during dropout.

- **Function**: `def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):`
- **Parameters**:
  - `Y`: true labels for the training data.
  - `weights`: dictionary containing the weights of the neural network.
  - `cache`: dictionary of outputs of each layer from forward propagation.
  - `alpha`: learning rate.
  - `keep_prob`: probability of keeping a neuron during dropout.
  - `L`: number of layers in the network.
- **Returns**: The function updates the weights after applying dropout.

### 6. `6-dropout_create_layer.py`
This script creates a **TensorFlow layer** that uses **dropout** during training. Dropout is implemented during training but is turned off during testing.

- **Function**: `def dropout_create_layer(prev, n, activation, keep_prob):`
- **Parameters**:
  - `prev`: tensor containing the output of the previous layer.
  - `n`: number of nodes in the layer.
  - `activation`: activation function to be used.
  - `keep_prob`: probability that a node will be kept.
- **Returns**: the output of the new layer using dropout.

### 7. `7-early_stopping.py`
This script implements **Early Stopping**. Early stopping halts training when the model’s validation performance stops improving, preventing overfitting by stopping the training at the optimal time.

- **Function**: `def early_stopping(cost, opt_cost, threshold, patience, count):`
- **Parameters**:
  - `cost`: current validation cost.
  - `opt_cost`: the lowest recorded validation cost.
  - `threshold`: threshold to determine significant improvement.
  - `patience`: the number of epochs to wait before stopping if no significant improvement is observed.
  - `count`: the current count of epochs with no improvement.
- **Returns**: A boolean indicating whether to stop training, followed by the updated count.

---

## Installation

To run these scripts, you'll need to install **TensorFlow** and other dependencies.

1. Set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install tensorflow numpy
```

3. Run the scripts:

Each Python script is self-contained. You can test the functionality by running the provided `main.py` files or integrating them into your neural network projects.

```bash
python3 0-l2_reg_cost.py
```