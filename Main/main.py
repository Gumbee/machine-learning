import time
import numpy as np
import Data.DataManager as DataManager
import Training.cost_model as cost_model

from matplotlib import pyplot as plt
from NeuralNetwork.neural_network import NeuralNetwork as NeuralNetwork
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer


def get_mean_correct(prediction, y):
    num_right = 0

    for i in range(0, len(prediction)):
        if np.array_equal(prediction[i], y[i]):
            num_right += 1

    print('Accuracy:', (num_right / (len(y) * 1.)) * 100)


def main():
    network = NeuralNetwork(400)
    network.add_hidden_layer(100)
    network.add_output_layer(10)

    X, y, X_val, y_val, X_test, y_test = DataManager.get_handwriting_data(0.8, 0.2)

    print('\nStarting timer...')
    t = time.time()
    network.train(X, y, 450, alpha=1.5, reg_lambda=0.5)
    # network.fmin(X, y, 1, 800)
    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)

    predictions = network.predict(X_val)

    confidence = network.feed_forward(X_val)

    print("\nVisualizing....\n")

    plt.ion()
    gray = DataManager.read_image('../Data/DataFiles/number2.png')
    DataManager.display_image(gray)

    prediction = network.predict(np.matrix(np.ravel(gray.T)))
    conf = network.feed_forward(np.matrix(np.ravel(gray.T)))

    print("\nPrediciton for zwei: ", (int(np.where(prediction[0] == 1)[0]) + 1) % 10,
          " with confidence:", conf[2][0][(int(np.where(prediction[0] == 1)[0]))])

    input_var = input('\rPress Enter to continue or \'q\' to exit....')

    DataManager.visualize(X_val, y_val, predictions, confidence)
    DataManager.visualize(network.model['weights'][0][:, 1:])


def linear_regression():
    init_theta = np.matrix([0, 0, 0]).astype(np.float64)

    X, y = DataManager.generate_data(100, 500, degree=3)

    optimizer = GradientDescentOptimizer(learning_rate=1e-13, reg_lambda=2e9)
    optimizer.train(init_theta, X, y, cost_model.sum_of_squares, cost_model.sum_of_squares_gradient, 10000)

    print("New Model:", init_theta)

    # print("Predictions:\n", X.dot(init_theta.T))

    DataManager.scatter(X, y, init_theta)

linear_regression()
