import numpy as np

from matplotlib import pyplot as plt


def display_image(img_matrix: np.matrix, size=1):
    """
    Takes a matrix of values and displays it as a grayscale image.

    :param img_matrix:  The matrix that is to be displayed as image
    :param size:        The size of the window
    :return:            None
    """
    plt.figure(figsize=(size, size))
    plt.imshow(img_matrix, interpolation='nearest', cmap='gray')
    plt.show()
    plt.pause(0.1)


def visualize_image(X: np.matrix, y: np.matrix = None, predictions=None, feed_values=None):
    """
    Takes a set and displays each test case as image (20x20) and displays the neural network's prediction.

    :param X:           The set that is to be visualized
    :param y:           (optional) The set's corresponding output.
    :param predictions: (optional) The set's prediction made by the neural network.
    :param feed_values: (optional) The feed forward values generated by the neural network when passing X as input.
    :return:            None
    """
    idx = np.random.permutation(len(X))

    i = 0
    plt.ion()
    input_key = ''
    while not input_key == 'q' and i < len(X):
        img = np.array(X[idx[i]]).reshape((20, 20)).T
        display_image(img)

        if predictions is not None:
            print('\n{:<14s}'.format('Prediction:'), '{:^2d}'.format((int(np.where(predictions[idx[i]] == 1)[0]) + 1) % 10))
        if y is not None:
            print('{:<14s}'.format('Value:'), '{:^2d}'.format((int(np.where(y[idx[i]] == 1)[0]) + 1) % 10))
        if feed_values is not None:
            print('Confidence:', feed_values[2][idx[i]][(int(np.where(predictions[idx[i]] == 1)[0]))])

        input_key = input('\rPress Enter to continue or \'q\' to exit....')
        plt.close()
        i += 1

    plt.ioff()


def visualize_training_step(theta, X, iteration, **kwargs):
    step = kwargs['step'] if 'step' in kwargs else 20
    if iteration % step == step-1:
        plt.plot(np.ravel(X[0:, 0].T), X.dot(theta.T), color='red', linewidth=0.5)
        plt.pause(0.2)


def visualize_final_result(X: np.matrix, y: np.matrix, theta=None, pause: float = 1.):
    """
    Takes a set X and a corresponding y and scatters the data. The first column of the set X acts as the x-axis values.
    Additionally a parameter vector theta can be passed on so that the hypothesis is also plotted.

    :param X:       A set X that is to be scattered (Shape: MxN)
    :param y:       A set y corresponding to the output of X (Shape: Mx1)
    :param theta:   (optional) A parameter vector (Shape: 1xN)
    :param pause:   (optional) Determines how long should be waited until the hypothesis is plotted 
    :return:        None
    """
    plt.ion()
    plt.scatter(np.ravel(X[0:, 0].T), np.ravel(y.T))
    plt.show()
    if theta is not None:
        plt.pause(pause)
        plt.plot(np.ravel(X[0:, 0].T), X.dot(theta.T), color='red')

    input_key = 'no_key'
    while input_key == 'no_key':
        plt.pause(0.05)
        input_key = input('\rPress any key to continue...')