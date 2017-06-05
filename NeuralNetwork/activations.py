import numpy as np

# ======== Sigmoid Function ========


def sigmoid():
    return sigmoid_activation, sigmoid_gradient, 'Sigmoid'


def sigmoid_activation(z):
    """
    The activation function. Takes an input z (can be a scalar, vector or matrix) and outputs a value between
    zero and one based on the input.

    :param z:   The input value
    :return:    A value (or vector/matrix of values) between 0 and 1
    """
    # np.clip is used to prevent overflowing
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))


def sigmoid_gradient(z):
    """
    The activation function's derivative function. Takes an input z (can be a scalar, vector or matrix) and outputs
    the sigmoid function's gradient value for that input.

    :param z:   The input value
    :return:    The sigmoid function's gradient for that input z
    """
    # np.clip is used to prevent overflowing
    return np.multiply(z, (1 - z))


# ======== Softmax Function ========
