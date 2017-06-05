import uuid
import numpy as np
import NeuralNetwork.activations as Activations

from Logging.logger import LogHandler as LogHandler
from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from parameters import GDLoggingParameters as GDLoggingParameters
from parameters import GradientDescentParameters as GradientDescentParameters


def ravel(weights: list):
    """
    Takes a list of weight matrices and converts them into one big vector.
    
    :param weights: The list of weights matrices
    :return:        A vector containing all elements of the weight matrices
    """
    b = []
    for matrix in weights:
        b = np.hstack((b, np.array(matrix).ravel()))
    return b


def unravel(vector: np.matrix, layers: list):
    """
    Takes a vector and reconstructs the weight matrices based on the given layer structure.
    
    :param vector:  The vector containing all weight elements
    :param layers:  The list of layers from which the weight matrices dimensions are reconstructed
    :return:        The weights as list of matrices
    """
    thetas = []

    current_pos = 0
    for i in range(1, len(layers)):
        m, n = layers[i].units, (layers[i - 1].units + 1)
        matrix = vector[current_pos:(current_pos + m*n)].reshape((m, n))
        current_pos += m*n
        thetas.append(matrix)

    return thetas


class NeuralNetwork(object):

    def __init__(self, input_units=5, EPSILON=0.12, name: str = None):
        print('Neural Network has', input_units, 'input units.')
        self.input_units = input_units
        self.has_output = False
        self.model = {'weights': [], 'layers': []}

        # create the input layer
        self.model['layers'].append(NeuralLayer(input_units))

        # EPSILON is used to initialize the random weights matrix
        self.EPSILON = EPSILON

        self.id = uuid.uuid4().hex

        self.name = name or 'Unnamed Neural Net'

    # ======== Network Operations ========

    def add_hidden_layer(self, units: int):
        """
        Adds a new hidden layer to the network. If there exists an output already, the new hidden layer is inserted
        between the current last hidden layer and the output.
        
        :param units:   The number of hidden units contained in this hidden layer
        :return:        None
        """
        # create new layer
        layer = NeuralLayer(units, True)

        if not self.has_output:
            # get previous layer
            prev_layer = self.model['layers'][len(self.model['layers']) - 1]
            # create random weights matrix
            weight = np.random.rand(units, prev_layer.units + (prev_layer.has_bias * 1)) * (2 * self.EPSILON) - self.EPSILON

            self.model['layers'].append(layer)
            self.model['weights'].append(weight)
        else:
            # get previous layer
            prev_layer = self.model['layers'][len(self.model['layers']) - 2]
            # create random weights matrix
            weight = np.random.rand(units, prev_layer.units + (prev_layer.has_bias * 1)) * (2 * self.EPSILON) - self.EPSILON

            self.model['layers'].insert(len(self.model['layers'])-1, layer)
            self.model['weights'].insert(len(self.model['weights'])-1, weight)

            output_layer = self.model['layers'][len(self.model['layers'])-1]

            # fix the last weight matrix (dimension could be wrong if we insert a layer before the end)
            self.model['weights'][len(self.model['weights'])-1] = np.random.rand(output_layer.units, units + 1) * (2*self.EPSILON) - self.EPSILON

        print('Added a new layer with ', units, ' units.')

    def add_output_layer(self, units: int):
        """
        Adds an output layer to the network. If a output layer exists already, that layer will be replaced with the new
        one. Any new hidden layers added to the network after an output layer was added will be inserted between the
        last hidden layer and the output layer.
        
        :param units:   The number of output units contained in this output layer
        :return:        None
        """
        # create new layer
        layer = NeuralLayer(units, False)

        if not self.has_output:
            # get previous layer
            prev_layer = self.model['layers'][len(self.model['layers']) - 1]
            # create random weights matrix
            weight = np.random.rand(units, prev_layer.units + (prev_layer.has_bias * 1)) * (2 * self.EPSILON) - self.EPSILON
            self.has_output = True
            self.model['layers'].append(layer)
            self.model['weights'].append(weight)

            print('Added a new output layer with ', units, ' units.')
        else:
            # get last hidden layer
            prev_layer = self.model['layers'][len(self.model['layers']) - 2]
            # create random weights matrix
            weight = np.random.rand(units, prev_layer.units + (prev_layer.has_bias * 1)) * (
            2 * self.EPSILON) - self.EPSILON
            # override existing values
            self.model['layers'][len(self.model['layers'])-1] = layer
            self.model['weights'][len(self.model['weights'])-1] = weight

            print('Output layer overridden with ', units, ' units.')

    # ======== Gradient Descent Functions ========

    def cost_wrapper(self, theta: np.matrix, X: np.matrix, y: np.matrix, reg_lambda=1.):
        """
        Wrapper function for scipy's optimization functions.
        
        :param theta:       One dimensional vector containing all weight elements of the network
        :param X:           The training set on which the network is trained
        :param y:           The training set's output
        :param reg_lambda:  (optional) Regularization term for the weights. Default: 1
        :return:            cost, gradients
        """

        weights = unravel(theta, self.model['layers'])

        feed_values = self.feed_forward(X, weights)

        J = self.cost_function(weights, X, y, reg_lambda, feed_values)
        gradients = ravel(self.gradient(weights, X, y, reg_lambda, feed_values))

        return J, gradients

    def cost_function(self, weights: list, X: np.matrix, y: np.matrix, reg_lambda=1., feed_values: np.array = None):
        """
        Evaluates the cost of the given weights based on a given training set X with output y. Optionally feed forward
        values can be passed on so that this method doesn't feed forward the training set again.
        
        :param weights:     the weights with which we compute the feed forward values
        :param X:           input on which the model is/was being trained
        :param y:           output corresponding to the input on which the model is/was being trained
        :param reg_lambda:  (optional) regularization term for the weights of the network. Default: 1
        :param feed_values: (optional) calculate the cost for the given feed_forward values
        :return:            cost of the model
        """
        layers = self.model['layers']

        m, n = X.shape

        L = len(layers)

        if feed_values is None:
            output = self.feed_forward(X)[L-1]
        else:
            output = feed_values[L-1]

        # calculate the first term needed for the cost function and lock the values between 1e-88 and 1 so we don't
        # get into a situation where we calculate log(0)
        first_term = np.clip(output, 1e-88, 1)
        # calculate the second term needed for the cost function and lock the values between 1e-88 and 1 so we don't
        # get into a situation where we calculate log(0)
        second_term = np.clip(np.subtract(1, output), 1e-88, 1)

        J = 0
        for j in range(0, layers[L-1].units):
            # add the error of every output unit based on it's desired output value
            J += (-y[:, j]).T.dot(np.log(first_term[:, j])) - (np.subtract(1, y[:, j])).T.dot(np.log(second_term[:, j]))

        J *= (1./m)

        for weight in weights:
            square = np.square(weight[:, 1:])
            # add the regularization term (squared weights minus the first column since we don't want
            # to penalize the bias units)
            J += (reg_lambda/(2.*m))*(np.sum(square))

        return J

    def gradient(self, weights: list, X: np.matrix, y: np.matrix, reg_lambda=1., feed_values: np.array = None):
        """
        Computes the gradients of the provided weight matrices.

        :param weights:     the weights whose gradient values will be calculated
        :param X:           The training set with which the network is trained
        :param y:           The training set's output
        :param reg_lambda:  (optional) Regularization term for the weights. Default: 1
        :param feed_values: (optional) calculate the cost for the given feed_forward values
        :return:            gradients of the weights
        """
        layers = self.model['layers']

        m, n = X.shape

        W = len(weights)
        L = len(layers)

        # check if the feed forward values are already provided or if we have to calculate them
        if feed_values is None:
            activations = self.feed_forward(X)
        else:
            activations = feed_values

        # allocate space
        deltas = [None] * L
        gradients = []

        # backpropagation part
        # calculate deltas (layer errors)
        deltas[L - 1] = (activations[L - 1] - y).T

        for i in range(L - 2, 0, -1):
            deltas[i] = np.multiply(weights[i].T.dot(deltas[i + 1])[1:, :], self.model['layers'][i].activation_gradient(activations[i].T))

        for i in range(0, W):
            # create a zero matrix with the same dimensions as our weight matrix
            sub = np.zeros_like(weights[i])
            # replace the first column of the zero matrix with the first column of the weight matrix
            sub[:, 0:1] = (reg_lambda / (m * 1.0)) * weights[i][:, 0:1]
            gradients.append((1. / m) * deltas[i + 1].dot(np.hstack((np.ones((m, 1)), activations[i]))))
            # add regularization term and subtract the first column
            gradients[i] += ((reg_lambda / (m * 1.0)) * weights[i]) - sub

        return gradients

    def train(self, X: np.matrix, y: np.matrix, Optimizer: callable(GradientDescentOptimizer), gd_params: GradientDescentParameters = None, log_handler: LogHandler = None):
        """
        Trains the network with gradient descent with the given training set and the corresponding output and
        applies the trained model to the network.
        
        :param X:           The training set with which the network is trained
        :param y:           The training set's output
        :param Optimizer:   The Optimizer to be used to optimize the cost function
        :param gd_params:   The parameters to be used for the training
        :param log_params:  The parameters to be used for the logging
        :return:            None
        """
        weights, layers = self.parse_model()

        gd_params = gd_params or GradientDescentParameters()

        gd_params.cost_func = self.cost_function
        gd_params.gradient_func = self.gradient

        # create an instance of GradientDescentOptimizer and optimize the weights
        optimizer = Optimizer()

        log_handler = log_handler or LogHandler()
        log_handler.gd_log_parameters.log_file_name = 'NeuralNets/' + log_handler.gd_log_parameters.log_file_name

        # if the log handler has already registered a (different) network, use a new log handler with the same settings
        # (but with a different file name)
        if 'network_info' in log_handler.log_dict and log_handler.log_dict['network_info']['id'] != self.id:
            new_log_handler = LogHandler()
            new_log_handler.gd_log_parameters.log_progress = log_handler.gd_log_parameters.log_progress
            new_log_handler.gd_log_parameters.num_cost_evaluations = log_handler.gd_log_parameters.num_cost_evaluations
            new_log_handler.gd_log_parameters.cost_eval_use_subset = log_handler.gd_log_parameters.cost_eval_use_subset
            new_log_handler.gd_log_parameters.cost_eval_subset_size = log_handler.gd_log_parameters.cost_eval_subset_size
            log_handler = new_log_handler

        optimizer.train(weights, X, y, gd_params, log_handler)

        log_handler.register_network(self.id, self.name, self.model)

        print(log_handler.log_dict)

    # ======== Helper Functions ========

    def feed_forward(self, X: np.matrix, weights: list = None):
        """
        Takes an input set X and optionally a list of weights, feeds the input forward and
        returns the activation values for every layer.
        
        :param X:       The input set which is fed forward
        :param weights: (optional) Weights which should be used for the forward propagation. Default: Current model's weights
        :return:        The activation values for every unit in every layer
        """
        m, n = X.shape

        _, layers = self.parse_model()

        if weights is None:
            weights, layers = self.parse_model()

        activations = [X]

        for i in range(0, len(layers)-1):
            X = np.hstack((np.ones((m, 1)), X))
            X = self.model['layers'][i+1].activate(np.dot(X, weights[i].T))
            activations.append(X)

        return activations

    def predict(self, X: np.matrix, threshold=0):
        """
        Takes an input set and predicts the output based on the current model. If a threshold is specified then every 
        output unit with a value greater than the threshold will output 1 and all else 0. If no threshold is specified
        then the output unit with the greatest value will output 1 and all other units will output 0.
        
        :param X:           The input set
        :param threshold:   (optional) The threshold which determines if a output unit outputs a 1 or 0. If the 
                            threshold is 0 then only the output unit with the greatest value will output 1.
        :return:            the prediction
        """
        weights, layers = self.parse_model()

        # only get the last layer as it contains our output
        output = self.feed_forward(X)[len(layers) - 1]

        for i in range(0, len(output)):
            if threshold > 0:
                # convert all outputs greater than the threshold to 1 and all else to 0
                idx = np.where(output[i] >= threshold)
            else:
                # only convert the output with the greatest value to 1 and all else to 0
                idx = np.where(output[i] == max(output[i]))

            output[i] = np.zeros_like(output[i])
            output[i][idx] = 1
        return output

    # ======== Util Functions ========

    def get_model(self):
        return self.model

    def parse_model(self):
        """
        Returns the current model's weights and layers
        
        :return: The model's weights and layers
        """
        weights = self.model['weights']
        layers = self.model['layers']

        return weights, layers



class NeuralLayer(object):
    """
    Class that contains necessary information for a neural layer (e.g how many units).
    """

    def __init__(self, units: int, has_bias=True, activator: callable = Activations.sigmoid):
        self.units = units
        self.has_bias = has_bias
        self.layer = np.zeros(units+(has_bias*1))

        self.activator, self.activator_gradient, self.activation_name = activator()

        # if this layer has a bias, we have to set the first item to 1
        if has_bias:
            self.layer[0] = 1

    def activate(self, z):
        """
        The activation function. Takes an input z (can be a scalar, vector or matrix) and outputs a value based on the input.

        :param z:   The input value
        :return:    A value (or vector/matrix of values)
        """
        if type(z).__module__ == np.__name__:
            return self.activator(z)
        else:
            return self.activator(np.array(z))

    def activation_gradient(self, z):
        """
        The activation function's derivative function. Takes an input z (can be a scalar, vector or matrix) and outputs
        the function's gradient value for that input.

        :param z:   The input value
        :return:    The activation function's gradient for that input z
        """
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return self.activator_gradient(z)
        else:
            return self.activator_gradient(np.array(z))

