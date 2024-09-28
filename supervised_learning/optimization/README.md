# Machine Learning Optimization Project

This project encompasses a series of tasks focused on implementing various machine learning techniques, including normalization, regularization, and optimization using TensorFlow. Each task is designed to build upon the previous one, ultimately culminating in the creation and training of a neural network model.

## Project Structure

```
.
├── 0-norm_constants.py
├── 1-normalization.py
├── 2-l2_regularization.py
├── 3-l1_regularization.py
├── 4-early_stopping.py
├── 5-batch_normalization.py
├── 6-dropout.py
├── 7-data_augmentation.py
├── 8-hyperparameter_tuning.py
├── 9-gradient_descent.py
├── 10-rmsprop.py
├── 11-adam_optimizer.py
├── 12-learning_rate_decay.py
├── 13-batch_normalization_tensorflow.py
├── 14-regularization_techniques.py
└── 15-model.py
```

## Task Descriptions

### 0. `norm_constants.py`
Defines the normalization constants used in various machine learning algorithms.

### 1. `normalization.py`
Implements normalization techniques for preparing data before training machine learning models.

### 2. `l2_regularization.py`
Implements L2 regularization for controlling model complexity and preventing overfitting.

### 3. `l1_regularization.py`
Implements L1 regularization, which promotes sparsity in the model weights.

### 4. `early_stopping.py`
Introduces early stopping to halt training when performance on a validation set starts to degrade.

### 5. `batch_normalization.py`
Implements batch normalization to improve the training speed and stability of the model.

### 6. `dropout.py`
Implements dropout as a regularization technique to reduce overfitting during training.

### 7. `data_augmentation.py`
Implements data augmentation techniques to artificially increase the size of the training dataset.

### 8. `hyperparameter_tuning.py`
Focuses on tuning hyperparameters to optimize model performance.

### 9. `gradient_descent.py`
Implements gradient descent algorithms for optimizing the loss function of the model.

### 10. `rmsprop.py`
Implements the RMSProp optimization algorithm to adjust the learning rate based on the average of recent magnitudes of the gradients.

### 11. `adam_optimizer.py`
Implements the Adam optimization algorithm, combining the advantages of AdaGrad and RMSProp.

### 12. `learning_rate_decay.py`
Implements techniques for decaying the learning rate over time to improve convergence.

### 13. `batch_normalization_tensorflow.py`
An implementation of batch normalization specifically utilizing TensorFlow.

### 14. `regularization_techniques.py`
Demonstrates the application of various regularization techniques in training models.

### 15. `model.py`
Builds, trains, and saves a neural network model using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization. 

## Dependencies

- Python 3.5 or higher
- TensorFlow 1.12
- Numpy 1.15

You can install the necessary packages via pip:

```bash
pip install numpy tensorflow==1.12
```

## Usage

To run the tasks, navigate to the directory containing the scripts and execute them as follows:

```bash
python 0-norm_constants.py
python 1-normalization.py
...
python 15-model.py
```

The scripts will run sequentially, utilizing the outputs of previous tasks where necessary.
