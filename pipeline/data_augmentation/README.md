# Data Augmentation Project

## Overview
This project implements various data augmentation techniques using TensorFlow for image processing. The implementation includes functions for flipping, cropping, rotating, shearing, and adjusting image properties like brightness and hue.

## Project Structure
```
pipeline/
└── data_augmentation/
    ├── README.md
    ├── 0-flip.py
    ├── 1-crop.py
    ├── 2-rotate.py
    ├── 3-shear.py
    ├── 4-brightness.py
    └── 5-hue.py
```

## Requirements
### General
* Ubuntu 16.04 LTS
* Python 3.6.12
* NumPy 1.16
* TensorFlow 1.15
* Allowed editors: vi, vim, emacs
* All files must end with a new line
* All files must be executable
* Code should follow pycodestyle style (version 2.4)

### Installation
```bash
# Install TensorFlow Datasets
pip install --user tensorflow-datasets
```

## Functions Implementation

### 1. Flip Image (0-flip.py)
```python
def flip_image(image)
```
* Flips an image horizontally
* Input: 3D tf.Tensor containing the image
* Output: Flipped image

### 2. Crop Image (1-crop.py)
```python
def crop_image(image, size)
```
* Performs a random crop of an image
* Input: 3D tf.Tensor and tuple containing crop size
* Output: Cropped image

### 3. Rotate Image (2-rotate.py)
```python
def rotate_image(image)
```
* Rotates an image by 90 degrees counter-clockwise
* Input: 3D tf.Tensor containing the image
* Output: Rotated image

### 4. Shear Image (3-shear.py)
```python
def shear_image(image, intensity)
```
* Randomly shears an image
* Input: 3D tf.Tensor and intensity value
* Output: Sheared image

### 5. Change Brightness (4-brightness.py)
```python
def change_brightness(image, max_delta)
```
* Randomly changes image brightness
* Input: 3D tf.Tensor and maximum delta value
* Output: Brightness-adjusted image

### 6. Change Hue (5-hue.py)
```python
def change_hue(image, delta)
```
* Changes the hue of an image
* Input: 3D tf.Tensor and delta value
* Output: Hue-adjusted image

## Usage Example
```python
#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Import function
flip_image = __import__('0-flip').flip_image

# Load dataset
doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)

# Apply augmentation and display
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(flip_image(image))
    plt.show()
```

## File Requirements
* First line of all files: `#!/usr/bin/env python3`
* Module documentation required
* Class documentation required
* Function documentation required
* Only allowed import: `import tensorflow as tf`

## Testing
Each function can be tested using its corresponding main file:
```bash
chmod +x 0-main.py
./0-main.py
```

## Author
[Maxime Guy Bakunzi](https://github.com/Maxime-Bakunzi/)
