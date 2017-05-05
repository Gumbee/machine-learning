import time
import numpy as np
import Data.DataManager as DataManager
import Utils.Visualizer as Visualizer
import Training.cost_model as cost_model

from NeuralNetwork.neural_network import NeuralNetwork as NeuralNetwork
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.gradient_descent import GradientDescentParameters as GradientDescentParameters


def get_mean_correct(prediction, y):
    num_right = 0

    for i in range(0, len(prediction)):
        if np.array_equal(prediction[i], y[i]):
            num_right += 1

    print('Accuracy:', (num_right / (len(y) * 1.)) * 100)


def main():
    network = NeuralNetwork(400)
    network.add_hidden_layer(400)
    network.add_output_layer(10)

    X, y, X_val, y_val, X_test, y_test = DataManager.get_handwriting_data(0.8, 0.2)

    print('\nStarting timer...')
    t = time.time()
    # network.train(X, y, 850, alpha=1.5, reg_lambda=0.5)
    network.fmin(X, y, reg_lambda=0.5)
    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)

    predictions = network.predict(X_val)

    confidence = network.feed_forward(X_val)

    print("\nVisualizing....\n")

    Visualizer.visualize_image(X_val, y_val, predictions, confidence)
    Visualizer.visualize_image(network.model['weights'][0][:, 1:])


def linear_regression():
    init_theta = np.matrix([0, 0, 0]).astype(np.float64)

    X, y = DataManager.generate_data(100, noise=20, degree=3)

    Visualizer.plt.ion()
    Visualizer.plt.scatter(np.ravel(X[0:, 0].T), np.ravel(y.T), s=12)
    Visualizer.plt.show()

    optimizer = GradientDescentOptimizer()

    # set gradient descent parameters
    gd_parameters = GradientDescentParameters()
    gd_parameters.learning_rate = 3e-11
    gd_parameters.reg_lambda = 0
    gd_parameters.cost_func = cost_model.sum_of_squares
    gd_parameters.gradient_func = cost_model.sum_of_squares_gradient
    gd_parameters.max_iter = 30000
    gd_parameters.callback = Visualizer.visualize_training_step
    gd_parameters.callback_args = {'step': 1000}

    # train
    optimizer.train(init_theta, X, y, gd_parameters)

    # Visualizer.visualize_final_result(X, y, init_theta)

linear_regression()
