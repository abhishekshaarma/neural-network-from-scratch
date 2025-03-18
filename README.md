# Neural Network Implementation from Scratch

A C++ implementation of a neural network built from scratch, optimized with the Eigen library for efficient matrix operations. (should I also upload my notes?)

## Overview

This project implements a feedforward neural network with customizable architecture for handwritten digit classification using the MNIST dataset(thank you for the dataset and for making it available). The implementation features gradient descent optimization with backpropagation for training.


## Dependencies

- C++ compiler supporting C++11 or later
- [Eigen](https://eigen.tuxfamily.org/) (3.4.0 or later) - Header-only library for matrix operations
- MNIST dataset files:
  - `train-images-idx3-ubyte` - Training images
  - `train-labels-idx1-ubyte` - Training labels
  - `t10k-images-idx3-ubyte` - Test images
  - `t10k-labels-idx1-ubyte` - Test labels

## Project Structure

```
neural-network/
├── main.cpp                # Main program entry point
├── layer.h                 # Neural network layer implementation
├── NeuralNetwork.h         # Neural network class
├── namespace.h             # Function namespace definitions
├── utils.h                 # Utility functions for MNIST data loading
└── README.md               # This file
```

## Implementation Details

### Layer Class

The `Layer` class represents a single layer in the neural network:
- Maintains weights, biases, and activation values
- Implements forward propagation
- Provides access to activation values and derivatives
- Handles weight updates during backpropagation

### Neural Network Class

The `NeuralNetwork` class orchestrates the layers:
- Manages multiple layer instances
- Implements forward and backward propagation
- Provides training and testing functionality
- Calculates loss during training

### Activation Functions

The network uses sigmoid activation functions:
- `sigmoid(x) = 1 / (1 + exp(-x))`
- `sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))`

## Usage

```cpp
// Create layers
Layer hidden_layer(784, 64, sigmoid, sigmoid_derivative);
Layer output_layer(64, 10, sigmoid, sigmoid_derivative);

// Construct neural network
NeuralNetwork nn({hidden_layer, output_layer});

// Train the network
nn.train(train_dataset, label_train_dataset, 0.01, 3);

// Test the network
nn.test(test_dataset, label_test_dataset);
```

## Example Results

The network achieves reasonable accuracy on the MNIST dataset after just a few epochs:
This output is for a smaller base because my vm kept taking alot of time and I have no patience.
![result](https://github.com/user-attachments/assets/2e7be963-5f41-4e31-b74a-f984d6a9dfbb)

