import numpy as np

# ======== Sigmoid Function ========


def sigmoid():
    return sigmoid_activation, sigmoid_gradient, 'Sigmoid'


def sigmoid_activation(z):
    """
    The activation function. Takes an input z (can be a scalar, vector or matrix) and outputs a value between
    zero and one based on the input.

    Args:
        z:          The input value
    
    Returns:
        np.array:   A value (or vector/matrix of values) between 0 and 1
    """
    # np.clip is used to prevent overflowing
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))


def sigmoid_gradient(z):
    """
    The activation function's derivative function. Takes an input z (can be a scalar, vector or matrix) and outputs
    the sigmoid function's gradient value for that input.

    Args:
        z:          The input value
    
    Returns:
        np.array:   The sigmoid function's gradient for that input z
    """
    s = sigmoid_activation(z)
    return np.multiply(s, (1 - s))


# ======== Softmax Function ========
