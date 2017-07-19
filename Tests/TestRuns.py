import time
import numpy as np
import Data.data_manager as DataManager
import Utils.visualizer as Visualizer
import Training.cost_model as cost_model

from NeuralNetwork.neural_network import NeuralNetwork as NeuralNetwork
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.adadelta import AdaDeltaOptimizer as AdaDeltaOptimizer
from Logging.logger import LogHandler as LogHandler
from Utils.anomaly_detector import AnomalyDetector as AnomalyDetector

from parameters import GradientDescentParameters as GradientDescentParameters


def get_mean_correct(prediction, y):
    num_right = 0

    for i in range(0, len(prediction)):
        if np.array_equal(prediction[i], y[i]):
            num_right += 1

    print('Accuracy:', (num_right / (len(y) * 1.)) * 100)


def neural_net_test(Optimizer: callable(GradientDescentOptimizer), batch_size: int = 60, epochs: int = 2, visualize: bool = False, network_name: str = None):
    network = NeuralNetwork(784, name=network_name)
    network.add_hidden_layer(100)
    network.add_hidden_layer(50)
    network.add_output_layer(10)

    gd_params = GradientDescentParameters()
    gd_params.learning_rate = 1
    gd_params.batch_size = batch_size
    gd_params.reg_lambda = 0
    gd_params.epochs = epochs

    X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

    t = time.time()
    network.train(X, y, Optimizer, gd_params)
    network.train(X, y, Optimizer, gd_params)
    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)
    get_mean_correct(network.predict(X_test), y_test)

    if visualize:
        predictions = network.predict(X_val)

        confidence = network.feed_forward(X_val)

        print("\nVisualizing....\n")

        Visualizer.visualize_image(X_val, y_val, size=28, transpose=False, predictions=predictions, feed_values=confidence)
        Visualizer.visualize_image(network.model['weights'][0][:, 1:], size=20)


def nn_optimizer_comparison(OptimizerA: callable(GradientDescentOptimizer), OptimizerB: callable(GradientDescentOptimizer), batch_size: int = 60, epochs: int = 2):
    network = NeuralNetwork(784)
    network.add_hidden_layer(500)
    network.add_hidden_layer(300)
    network.add_output_layer(10)

    gd_params = GradientDescentParameters()
    gd_params.learning_rate = 1
    gd_params.batch_size = batch_size
    gd_params.reg_lambda = 0
    gd_params.epochs = epochs

    log_handler = LogHandler()

    X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

    print()
    print(OptimizerA.__name__, "  -------------------------------------")

    t = time.time()

    init_val = [np.matrix(x) for x in network.model['weights']]
    network.train(X, y, OptimizerA, gd_params, log_handler)

    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)
    get_mean_correct(network.predict(X_test), y_test)

    network.model['weights'] = init_val

    print()
    print(OptimizerB.__name__, "  -------------------------------------")

    t = time.time()

    network.train(X, y, OptimizerB, gd_params, log_handler)

    t = time.time() - t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)
    get_mean_correct(network.predict(X_test), y_test)


def linear_regression_test(Optimizer: callable(GradientDescentOptimizer) = GradientDescentOptimizer):
    init_theta = np.matrix([0, 0, 0]).astype(np.float64)

    X, y = DataManager.generate_data(1000, noise=10, degree=3)

    Visualizer.plt.ion()
    Visualizer.plt.scatter(np.ravel(X[0:, 0].T), np.ravel(y.T), s=12)
    Visualizer.plt.show()

    optimizer = Optimizer()

    # set gradient descent parameters
    gd_parameters = GradientDescentParameters()
    gd_parameters.learning_rate = 3e-17
    gd_parameters.epochs = 50
    gd_parameters.batch_size = 1000
    gd_parameters.reg_lambda = 0
    gd_parameters.cost_func = cost_model.sum_of_squares
    gd_parameters.gradient_func = cost_model.sum_of_squares_gradient
    gd_parameters.callback = Visualizer.visualize_training_step
    gd_parameters.callback_args = {'step': 1}

    # train
    optimizer.train(init_theta, X, y, gd_parameters)

    # Visualizer.visualize_final_result(X, y, init_theta)


def anomaly_test():

    detector = AnomalyDetector(multivariate=True)

    noise = 20000000000

    X, y = DataManager.generate_data(400, noise=noise, degree=6)
    thetas = np.matrix(np.ones_like(X[0, :])*0.01)
    thetas[0, 2] = 100000
    thetas[0, 4] = - 2

    y = np.random.normal(X.dot(thetas.T), noise)

    x = np.hstack((X, y))

    detector.train(x)

    choice = 30

    idx = np.random.choice(len(y), choice)

    y[idx] = np.random.normal(y[idx], 4*noise)

    x = np.hstack((X, y))

    detector.visualize_anomalies(x, scatter=False)
