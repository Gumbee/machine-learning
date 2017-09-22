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

The following guide steps you through the process of creating a neural network and visualizing the performance and data. The network will be trained to recognize digits (10 classes, 0-9) using the MNIST data set. For a more detailed guide (also on how to visualize the data, performance etc.) take a look at the [NotSoQuickGuide](https://github.com/Gumbee/machine-learning/blob/master/NotSoQuickGuide.md).

```Python
import Data.data_manager as DataManager
from NeuralNetwork.neural_network import NeuralNetwork
from Training.gradient_descent import GradientDescentOptimizer

# create a neural network with 784 input units and some hidden layers
network = NeuralNetwork(784, name="Example Network")
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_hidden_layer(100)
network.add_output_layer(10)

# get the data (data not included, download the data yourself and move it to the Data/DataFiles folder)
# data can be downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
X, y, X_val, y_val, X_test, y_test = DataManager.get_mnist_data()

# train the network with the gradient descent optimizer and the specified settings
network.train(X, y, GradientDescentOptimizer)

print('Accuracy Training Set: ', network.get_mean_correct(X, y))
print('Accuracy Validation Set: ', network.get_mean_correct(X_val, y_val))
```

## Board Screenshots
##### Network Page
![Board](http://i.imgur.com/GQ1tQY3.png)
##### Dataset Visualization
![Dataset](http://i.imgur.com/1cLafro.png)

## License
MIT License

Copyright (c) 2017 Mugeeb Al-Rahman Hassan