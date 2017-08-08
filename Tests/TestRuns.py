import Data.data_manager as DataManager
import numpy as np
import time
import Training.cost_model as cost_model
import Utils.visualizer as Visualizer

from Logging.logger import LogHandler
from NeuralNetwork.neural_network import NeuralNetwork
from Training.gradient_descent import GradientDescentOptimizer
from Utils.anomaly_detector import AnomalyDetector

from parameters import GradientDescentParameters as GradientDescentParameters


def neural_net_test(Optimizer: callable(GradientDescentOptimizer), batch_size: int = 60, epochs: int = 2, visualize: bool = False, network_name: str = None):
    # define network architecture
    network = NeuralNetwork(784, name=network_name)
    network.add_hidden_layer(200)
    network.add_hidden_layer(200)
    network.add_hidden_layer(50)
    network.add_output_layer(10)

    # set parameters
    gd_params = GradientDescentParameters()
    gd_params.learning_rate = 0.2
    gd_params.batch_size = batch_size
    gd_params.reg_lambda = 0
    gd_params.epochs = epochs

    # get the data sets
    X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

    # create a log handler who will handle the logging of the progress
    log_handler = LogHandler()

    # add some data that we want to visualize
    log_handler.add_data_set(X, y, subset_size=5000)
    log_handler.add_data_set(X_val, y_val, subset_size=5000)
    log_handler.add_data_set(X_test, y_test, subset_size=5000)

    # add a few data sets on which we want to monitor the network's accuracy to the log handler
    log_handler.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
    log_handler.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)
    log_handler.add_accuracy_monitor(X_test, y_test, name="Test Set", subset_size=500)

    # train the network and output the duration it took to train the net
    t = time.time()
    network.train(X, y, Optimizer, gd_params, log_handler)
    network.train(X, y, Optimizer, gd_params, log_handler)
    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    # display the final accuracies on each data set
    print('Accuracy Training: ', network.get_mean_correct(X, y))
    print('Accuracy Validation: ', network.get_mean_correct(X_val, y_val))
    print('Accuracy Test: ', network.get_mean_correct(X_test, y_test))

    # visualize results if desired
    if visualize:
        predictions = network.predict(X_val)

        confidence = network.feed_forward(X_val)

        print("\nVisualizing....\n")

        Visualizer.visualize_image(X_val, y_val, size=28, transpose=False, predictions=predictions, feed_values=confidence)
        Visualizer.visualize_image(network.model['weights'][0][:, 1:], size=20)


def nn_optimizer_comparison(OptimizerA: callable(GradientDescentOptimizer), OptimizerB: callable(GradientDescentOptimizer), batch_size: int = 60, epochs: int = 2):
    # define the network architecture
    network = NeuralNetwork(784)
    network.add_hidden_layer(500)
    network.add_hidden_layer(300)
    network.add_output_layer(10)

    # set the training parameters
    gd_params = GradientDescentParameters()
    gd_params.learning_rate = 1
    gd_params.batch_size = batch_size
    gd_params.reg_lambda = 0
    gd_params.epochs = epochs

    X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

    # create a new log handler
    log_handler = LogHandler()

    log_handler.add_data_set(X, y, subset_size=5000)
    log_handler.add_data_set(X_val, y_val, subset_size=5000)
    log_handler.add_data_set(X_test, y_test, subset_size=5000)

    # add a few data sets on which we want to monitor the network's accuracy to the log handler
    log_handler.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
    log_handler.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)
    log_handler.add_accuracy_monitor(X_test, y_test, name="Test Set", subset_size=500)

    print()
    print(OptimizerA.__name__, "  -------------------------------------")

    t = time.time()

    # save the network's initial weights so that we can copy these weights
    # to the second network so that they have the same starting conditions
    init_val = [np.matrix(x) for x in network.model['weights']]
    # train the net with the first optimizer
    network.train(X, y, OptimizerA, gd_params, log_handler)

    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    # display the accuracy
    print(network.get_mean_correct(X, y))
    print(network.get_mean_correct(X_val, y_val))
    print(network.get_mean_correct(X_test, y_test))

    # copy the initial weights to the network
    network.model['weights'] = init_val

    print()
    print(OptimizerB.__name__, "  -------------------------------------")

    t = time.time()

    # train the net with the second optimizer
    network.train(X, y, OptimizerB, gd_params, log_handler)

    t = time.time() - t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    # display the accuracy
    print(network.get_mean_correct(X, y))
    print(network.get_mean_correct(X_val, y_val))
    print(network.get_mean_correct(X_test, y_test))


def linear_regression_test(Optimizer: callable(GradientDescentOptimizer) = GradientDescentOptimizer):
    # create a weight matrix whose values we will learn
    init_theta = np.matrix([0, 0, 0]).astype(np.float64)

    # create some random polynomial data
    X, y = DataManager.generate_data(1000, noise=100000, degree=3)

    # plot the data
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
    # create a multivariate anomaly detector
    detector = AnomalyDetector(multivariate=True)

    # define the noise range
    noise = 20000000000

    # create some random polynomial data
    X, y = DataManager.generate_data(400, noise=noise, degree=6)

    # create a coefficient vector (a,b,c,d,e in ax+bx^2....)
    thetas = np.matrix(np.ones_like(X[0, :])*0.01)
    # define some not so random values for some of the parameters so that we get 'nice looking' data
    thetas[0, 2] = 100000
    thetas[0, 4] = - 2

    # multiply the parameters with the polynomial input data
    y = np.random.normal(X.dot(thetas.T), noise)
    # stack the input data and the output data so that we can feed it into the detector
    x = np.hstack((X, y))

    # train the detector
    detector.train(x)
    # define some threshold for anomalies (this should usually be done based on some actual
    # anomalies, not the way we do it here!)
    detector.epsilon = np.min(detector.hypothesis(x))*5

    # choose 30 random points
    choice = 30
    idx = np.random.choice(len(y), choice)

    # manually create some anomalies
    y[idx] = np.random.normal(y[idx], 4*noise)
    # stack the data so that we can feed it into the detector
    x = np.hstack((X, y))

    # visualize the anomalies
    detector.visualize_anomalies(x, scatter=False)
