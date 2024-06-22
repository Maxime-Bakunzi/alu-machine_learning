
# Bayesian Probability

This project focuses on implementing various Bayesian probability calculations using Python. The code is compliant with `pycodestyle` and is designed to work with Python 3.5 and `numpy` 1.15 on Ubuntu 16.04 LTS.

## Project Structure

- `0-likelihood.py`: Contains the function to calculate the likelihood of observing data given various hypothetical probabilities.
- `1-intersection.py`: Contains the function to calculate the intersection of obtaining data with various hypothetical probabilities given prior beliefs.
- `2-marginal.py`: Contains the function to calculate the marginal probability of obtaining the data.
- `3-posterior.py`: Contains the function to calculate the posterior probability for the various hypothetical probabilities of developing severe side effects given the data.
- `README.md`: Provides an overview of the project.

## Requirements

- Python 3.5
- numpy 1.15
- pycodestyle 2.5

## Functions

### Likelihood

**File:** `0-likelihood.py`

Calculates the likelihood of obtaining data given various hypothetical probabilities.

```python
def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data given various hypothetical
    probabilities of developing severe side effects.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical probabilities
                       of developing severe side effects.

    Returns:
    numpy.ndarray: 1D array containing the likelihood of obtaining the data, x and n,
                   for each probability in P, respectively.
    """
```

### Intersection

**File:** `1-intersection.py`

Calculates the intersection of obtaining data with various hypothetical probabilities given prior beliefs.

```python
def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with the various hypothetical
    probabilities.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical probabilities
                       of developing severe side effects.
    Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    numpy.ndarray: 1D array containing the intersection of obtaining x and n with
                   each probability in P, respectively.
    """
```

### Marginal Probability

**File:** `2-marginal.py`

Calculates the marginal probability of obtaining the data.

```python
def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical probabilities
                       of developing severe side effects.
    Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    float: The marginal probability of obtaining x and n.
    """
```

### Posterior Probability

**File:** `3-posterior.py`

Calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data.

```python
def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical probabilities
    of developing severe side effects given the data.

    Parameters:
    x (int): The number of patients that develop severe side effects.
    n (int): The total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical probabilities
                       of developing severe side effects.
    Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    numpy.ndarray: 1D array containing the posterior probability of each probability
                   in P given x and n, respectively.
    """
```

## Usage

To use these functions, you can import them into your Python script as follows:

```python
#!/usr/bin/env python3

import numpy as np
from posterior import posterior

P = np.linspace(0, 1, 11)
Pr = np.ones(11) / 11
print(posterior(26, 130, P, Pr))
```

### Example

Here is an example of how to use the `posterior` function:

```python
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
```

## Validation and Testing

To validate and test the implementation, you can use the provided `main.py` files for each task. For example, to test the `posterior` function:

```bash
$ ./3-main.py
```
