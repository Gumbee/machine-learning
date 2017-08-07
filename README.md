# Machine Learning Framework
## Getting started

#### Installation

1. Fork the project
2. Make sure you've installed following dependencies:
    - Numpy
    - Flask
    - Pickle
    - Matplotlib
3. Done.

#### Quick guide

The following snippet lets you test out the framework with a simple example:
```Python
import Data.data_manager as DataManager
from Training.gradient_descent import GradientDescentOptimizer

# create a neural network with 784 input units and some hidden layers
network = NeuralNetwork(784, name="Example Network")
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_output_layer(10)

# get the data (data not included, download the data yourself and move it to the Data/DataFiles folder)
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

# train the network with the gradient descent optimizer and the specified settings
network.train(X, y, GradientDescentOptimizer())

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

The following snippet shows how to visualize the performance and the data using the LogHandler:
```Python
import Data.data_manager as DataManager

from Logging.logger import LogHandler as LogHandler
from Training.gradient_descent import GradientDescentOptimizer

# create a neural network with 784 input units and some hidden layers
network = NeuralNetwork(784, name="Example Network")
network.add_hidden_layer(200)
network.add_hidden_layer(200)
network.add_hidden_layer(50)
network.add_output_layer(10)

# get the data (data not included, download the data yourself and move it to the Data/DataFiles folder)
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

# create a log handler so we can visualize our progress
log_handler = LogHandler()

# add the training set to the log handler so that we can see a visual interpretation of the data set
log_handler.add_data_set(X, y, subset_size=5000)

# add a few data sets whose accuracy performance we want to monitor to the log handler
log_handler.gd_log_parameters.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
log_handler.gd_log_parameters.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)

# set some gradient descent parameters
gd_params = GradientDescentParameters()
gd_params.learning_rate = 0.2
gd_params.epochs = 2

# train the network with the gradient descent optimizer and the specified settings
network.train(X, y, GradientDescentOptimizer(), gd_params, log_handler)

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

## License
MIT License

Copyright (c) 2017 Mugeeb Al-Rahman Hassan
