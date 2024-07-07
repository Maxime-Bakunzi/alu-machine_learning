# Convolutions and Pooling in Python

This project demonstrates various operations in image processing using convolutional and pooling techniques. These operations are fundamental in many computer vision tasks, particularly in deep learning models such as Convolutional Neural Networks (CNNs).

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [Convolve Grayscale](#convolve-grayscale)
  - [Convolve Grayscale with Padding](#convolve-grayscale-with-padding)
  - [Convolve Grayscale with Stride](#convolve-grayscale-with-stride)
  - [Convolve Channels](#convolve-channels)
  - [Convolve Multiple Kernels](#convolve-multiple-kernels)
  - [Pooling](#pooling)
- [Examples](#examples)
- [License](#license)

## Introduction

This project contains functions to perform convolutions and pooling operations on images. The main operations included are:
- Valid Convolution on grayscale images
- Same Convolution on grayscale images
- Convolution on grayscale images with custom padding
- Convolution on grayscale images with custom stride
- Convolution with channels
- Convolution with multiple kernels
- Max and average pooling

## Prerequisites

- Python 3.5
- NumPy 1.15
- Matplotlib 

## Installation

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/Maxime-Bakunzi/alu-machine_learning.git
cd alu-machine_learning/math/convolutions_and_pooling
```

## Usage

### Running the Scripts

To run the example scripts, use the following commands:

```sh
python3 0-main.py
python3 1-main.py
python3 2-main.py
python3 3-main.py
python3 4-main.py
python3 5-main.py
python3 6-main.py
```

### Dataset

The project uses datasets one which is called MNIST.npz and the other of animal images stored in a NumPy `.npz` file. Make sure the datasets are located in the correct path.

```sh
dataset = np.load('../../supervised_learning/data/MNIST.npz')
```
Or
```sh
dataset = np.load('../../supervised_learning/data/animals_1.npz')
```

## Functions

### Valid Convolution

Performs a valid convolution on grayscale images.

#### Function Definition

```python
def convolve_grayscale_valid(images, kernel):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w)` containing multiple grayscale images.
- `kernel`: `numpy.ndarray` of shape `(kh, kw)` containing the kernel for the convolution.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Same Convolution

Performs a same convolution on grayscale images.

#### Function Definition

```python
def convolve_grayscale_same(images, kernel):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w)` containing multiple grayscale images.
- `kernel`: `numpy.ndarray` of shape `(kh, kw)` containing the kernel for the convolution.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Convolve with Padding

Performs a convolution on grayscale images with custom padding.

#### Function Definition

```python
def convolve_grayscale_padding(images, kernel, padding):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w)` containing multiple grayscale images.
- `kernel`: `numpy.ndarray` of shape `(kh, kw)` containing the kernel for the convolution.
- `padding`: tuple of `(ph, pw)` representing padding for the height and width.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Strided Convolution

Performs a convolution on grayscale images.

#### Function Definition

```python
def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w)` containing multiple grayscale images.
- `kernel`: `numpy.ndarray` of shape `(kh, kw)` containing the kernel for the convolution.
- `padding`: either a tuple `(ph, pw)`, `'same'`, or `'valid'`.
- `stride`: tuple of `(sh, sw)` representing stride for the height and width.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Convolve with Channels

Performs a convolution on images with channels.

#### Function Definition

```python
def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w, c)` containing multiple images.
- `kernel`: `numpy.ndarray` of shape `(kh, kw, c)` containing the kernel for the convolution.
- `padding`: either a tuple `(ph, pw)`, `'same'`, or `'valid'`.
- `stride`: tuple `(sh, sw)`.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Convolve Multiple Kernels

Performs a convolution on images using multiple kernels.

#### Function Definition

```python
def convolve(images, kernels, padding='same', stride=(1, 1)):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w, c)` containing multiple images.
- `kernels`: `numpy.ndarray` of shape `(kh, kw, c, nc)` containing the kernels for the convolution.
- `padding`: either a tuple `(ph, pw)`, `'same'`, or `'valid'`.
- `stride`: tuple `(sh, sw)`.

#### Returns

- `numpy.ndarray` containing the convolved images.

### Pooling

Performs pooling on images.

#### Function Definition

```python
def pool(images, kernel_shape, stride, mode='max'):
    # Implementation here
```

#### Parameters

- `images`: `numpy.ndarray` of shape `(m, h, w, c)` containing multiple images.
- `kernel_shape`: tuple `(kh, kw)` containing the kernel shape for the pooling.
- `stride`: tuple `(sh, sw)`.
- `mode`: indicates the type of pooling (`'max'` or `'avg'`).

#### Returns

- `numpy.ndarray` containing the pooled images.

## Examples

### Valid Convolution

```python
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
```

### Same Convolution

```python
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
```

### Convolution with Padding

```python
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
```

### Strided Convolution

```python
import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
```

### Convolution with Channels

```python
import numpy as np
import matplotlib.pyplot as plt
convolve_channels = __import__('4-convolve_channels').convolve_channels

if __name__ == '__main__':
    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], 
                       [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], 
                       [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
    images_conv = convolve_channels(images, kernel, padding='valid')
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0])
    plt.show()
```

### Convolution with Multiple Kernels

```python
import matplotlib.pyplot as plt
import numpy as np
convolve = __import__('5-convolve').convolve


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                       [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                       [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])

    images_conv = convolve(images, kernels, padding='valid')
    print(images_conv.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 0])
    plt.show()
    plt.imshow(images_conv[0, :, :, 1])
    plt.show()
    plt.imshow(images_conv[0, :, :, 2])
    plt.show()
```

### Pooling

```python
import numpy as np
import matplotlib.pyplot as plt
pool = __import__('6-pool').pool

if __name__ == '__main__':
    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_pool[0] / 255)
    plt.show()
```
