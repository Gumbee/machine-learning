# import a data manager who handles getting the MNIST data for us
import Data.data_manager as DataManager
# import a gradient descent optimizer so we can train the network with gradient descent
from Training.gradient_descent import GradientDescentOptimizer
# import the neural network class
from NeuralNetwork.neural_network import NeuralNetwork
from Logging.logger import LogHandler as LogHandler

# create a neural network with 784 input units and some hidden layers
network = NeuralNetwork(784, name="Example Network")
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_output_layer(10)

# get the data (data not included, download the data yourself and move it to the Data/DataFiles folder)
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

# train the network with the gradient descent optimizer
network.train(X, y, GradientDescentOptimizer)

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))

# create a log handler so we can visualize our progress
log_handler = LogHandler()

# add the training set to the log handler so that we can see a visual interpretation of the data set
log_handler.add_data_set(X, y, subset_size=5000)
# add the cross validation set to the log handler so that we can see a visual interpretation of the data set
log_handler.add_data_set(X_val, y_val, subset_size=5000)

# add a few data sets whose accuracy performance we want to monitor and visualize on a graph
# make sure to add these lines before training the network so that we actually capture the
# performance change during the training
log_handler.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
log_handler.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)

# train the network with the gradient descent optimizer and the log handler
network.train(X, y, GradientDescentOptimizer, log_handler=log_handler)

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))