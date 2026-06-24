# Neutronix

Neutronix is a simple-to-use ML library built from scratch using NumPy.

## Package Architecture

The library is organized into three core subpackages:

### 1. `optimizers`
Currently contains two optimizers:
* `GradientDescent`
* `GradientAscent`

### 2. `supervised`
Contains many of the classical machine learning models:
* `Linear Regression`
* `Gaussian Discriminative Analysis` (GDA)
* `Locally Weighted Linear Regression` (LWLR)
* `Naive Bayes`
* `Perceptron`
* `Softmax Regression`

### 3. `unsupervised`
Contains unsupervised learning and clustering models:
* `K-Means Clustering`
* `Mixtures of Gaussian Model` (GMM)
* `Principal Component Analysis` (PCA)
* `Factor Analysis`

---

This is a modular implementation of the models taught in the **CS229 (Autumn 2018)** batch by Andrew Ng.

## Installation

To install the package locally in editable mode:

```bash
pip install -e .
```

## Usage Example

Here is a quick example showing how to import and use a supervised model with an optimizer:

```python
import numpy as np
from neutronix import LinearRegression
from neutronix.optimizers import GradientDescent

# Prepare dummy data
X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_train = np.array([400, 450, 520, 580, 640], dtype=float)

# Initialize the model and attach the optimizer
model = LinearRegression()
model.optimizer = GradientDescent(alpha=0.01).batch_gradient_descent

# Train the model
model.fit(X_train, y_train, epoch=1000, verbose=True)

# Make predictions
predictions = model.predict(X_train)
```

## To-Do List

- [ ] Make it more modular by creating subclasses for `load` and `save` inheriting them
- [ ] Add more optimizers
- [ ] Unstubbing the loss function

---

## Repository History

This is the original **learning-ml** repository, which was later refactored and restructured with a Common Translation API into a usable library.

**GitHub Repository:** [learning-ml](https://github.com/Nirusaki-Malaal/learning-ml)