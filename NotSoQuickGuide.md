# Not So Quick Guide

The following guide steps you through the process of creating a neural network and visualizing the performance and data. The network will be trained to recognize digits (10 classes, 0-9) using the MNIST data set.

#### Imports
```Python
# import a data manager who handles getting the MNIST data for us
import Data.data_manager as DataManager
# import a gradient descent optimizer so we can train the network with gradient descent
from Training.gradient_descent import GradientDescentOptimizer
# import the neural network class
from NeuralNetwork.neural_network import NeuralNetwork
```

#### Creating the network
```Python
# create a neural network with 784 input units and some hidden layers
network = NeuralNetwork(784, name="Example Network")
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_output_layer(10)
```

#### Getting the data
```Python
# get the data (data not included, download the data yourself and move it to the Data/DataFiles folder)
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()
```

#### Training the network
```Python
# train the network with the gradient descent optimizer
network.train(X, y, GradientDescentOptimizer)
```

#### Outputting the performance
```Python
print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

The full code right now looks like this:
```Python
# import a data manager who handles getting the MNIST data for us
import Data.data_manager as DataManager
# import a gradient descent optimizer so we can train the network with gradient descent
from Training.gradient_descent import GradientDescentOptimizer
# import the neural network class
from NeuralNetwork.neural_network import NeuralNetwork

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
```

## Adding visualization

To add visual representations of the performance (accuracy) and the data we need to create a log handler which handles all of that for us. The log handler logs the progress and the data set to a file and allows the web interface to visualize these aspects of our machine learning system. First, we need to extend our imports:

#### Extending the imports
```Python
from Logging.logger import LogHandler as LogHandler
```

#### Creating the log handler
```Python
# create a log handler so we can visualize our progress
log_handler = LogHandler()
```

#### Adding the data set
```Python
# add the training set to the log handler so that we can see a visual interpretation of the data set
log_handler.add_data_set(X, y, subset_size=5000)
# add the cross validation set to the log handler so that we can see a visual interpretation of the data set
log_handler.add_data_set(X_val, y_val, subset_size=5000)
```

#### Monitor the performance
```Python
# add a few data sets whose accuracy performance we want to monitor and visualize on a graph
# make sure to add these lines before training the network so that we actually capture the
# performance change during the training
log_handler.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
log_handler.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)
```

Now that we have created a log handler, lets train the network again and give it that log handler so that we can visualize the desired aspects of our system:
```Python
# train the network with the gradient descent optimizer and the log handler
network.train(X, y, GradientDescentOptimizer, log_handler=log_handler)

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

The full code right now looks like this:

```Python
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
```

In the last example we trained the network once without monitoring the network's performance over the data sets and then we create the log handler and train the network again and monitor the network's performance over the data sets.

## Viewing the visualization

To have a look at the visualizations we need to start up our web interface (board). To do that, open the command line and navigate to the project's directory:

```
cd project_directory
```
Then run the board script:
```
python3 Board/board.py
```

To view the visualizations go to the tab ```Networks``` and click on the card named "Example Network". You should now see the overview page of your network.