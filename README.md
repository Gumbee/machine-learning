# Machine Learning Framework
## Getting started

#### Installation

1. Fork the project
2. Make sure you've installed following dependencies:
    - NumPy
    - Flask
    - Matplotlib
3. Done.

## Quick guide

The following guide steps you through the process of creating a neural network and visualizing the performance and data. The network will be trained to recognize digits (10 classes, 0-9) using the MNIST data set.

#### Imports
```Python
# import a data manager who handles getting the MNIST data for us
import Data.data_manager as DataManager
# import a gradient descent optimizer so we can train the network with gradient descent
from Training.gradient_descent import GradientDescentOptimizer
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
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()
```

#### Training the network
```Python
# train the network with the gradient descent optimizer
network.train(X, y, GradientDescentOptimizer())
```

#### Outputting the performance
```Python
print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

### Adding visualization

To add visual representations of the performance (accuracy) and the data we need to create a log handler which handles all of that for us. First, we need to extend our imports:

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
# make sure to add these lines before training the network so that we actually capture the
# performance change during the training
log_handler.add_accuracy_monitor(X, y, name="Training Set", subset_size=500)
log_handler.add_accuracy_monitor(X_val, y_val, name="Validation Set", subset_size=500)
```

## License
MIT License

Copyright (c) 2017 Mugeeb Al-Rahman Hassan