import numpy as np
import scipy.optimize as opt

from Training.gradient_descent import GradientDescentOptimizer as GradientDescentOptimizer
from Training.gradient_descent import GradientDescentParameters as GradientDescentParameters


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

    def __init__(self, input_units=5, EPSILON=0.12):
        print('Neural Network has', input_units, 'input units.')
        self.input_units = input_units
        self.has_output = False
        self.model = {'weights': [], 'layers': []}

        # create the input layer
        self.model['layers'].append(NeuralLayer(input_units))

        # EPSILON is used to initialize the random weights matrix
        self.EPSILON = EPSILON

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

        # calculate deltas (layer errors)
        deltas[L - 1] = (activations[L - 1] - y).T

        for i in range(L - 2, 0, -1):
            deltas[i] = np.multiply(weights[i].T.dot(deltas[i + 1])[1:, :], self.sigmoid_gradient(activations[i].T))

        for i in range(0, W):
            # create a zero matrix with the same dimensions as our weight matrix
            sub = np.zeros_like(weights[i])
            # replace the first column of the zero matrix with the first column of the weight matrix
            sub[:, 0:1] = (reg_lambda / (m * 1.0)) * weights[i][:, 0:1]
            gradients.append((1. / m) * deltas[i + 1].dot(np.hstack((np.ones((m, 1)), activations[i]))))
            # add regularization term and subtract the first column
            gradients[i] += ((reg_lambda / (m * 1.0)) * weights[i]) - sub

        return gradients

    def fmin(self, X: np.matrix, y: np.matrix, reg_lambda=1, max_iter=600):
        """
        Trains the network with scipy's opt function with the given training set and the corresponding output and
        applies the trained model to the network.
        
        :param X:           The training set with which the network is trained
        :param y:           The training set's output
        :param reg_lambda:  (optional) Regularization term for the weights. Default: 1
        :param max_iter:    (optional) Maximal number of iterations before the function should end the training
        :return:            None
        """
        # get optimized result from opt.fmin
        result = opt.fmin_tnc(func=self.cost_wrapper, x0=ravel(self.model['weights']), args=(X, y, reg_lambda), maxfun=max_iter)

        self.model['weights'] = unravel(result[0], self.model['layers'])

    def train(self, X: np.matrix, y: np.matrix, max_iter=5000, alpha=0.1, reg_lambda=1, debug_mode=True):
        """
        Trains the network with gradient descent with the given training set and the corresponding output and
        applies the trained model to the network.
        
        :param X:           The training set with which the network is trained
        :param y:           The training set's output
        :param max_iter:    (optional) Maximal number of iterations before the function should end the training
        :param alpha:       (optional) Learning Rate of the gradient descent algorithm. Default: 0.1
        :param reg_lambda:  (optional) Regularization term for the weights. Default: 1
        :param debug_mode:  (optional) True if debug mode should be turned on (outputs a table with important values)
        :return:            None
        """
        weights, layers = self.parse_model()

        # create an instance of GradientDescentOptimizer and optimize the weights
        optimizer = GradientDescentOptimizer()

        gd_parameters = GradientDescentParameters()
        gd_parameters.learning_rate = alpha
        gd_parameters.reg_lambda = reg_lambda
        gd_parameters.cost_func = self.cost_function
        gd_parameters.gradient_func = self.gradient
        gd_parameters.max_iter = max_iter
        gd_parameters.debug_mode = debug_mode

        optimizer.train(weights, X, y, gd_parameters)

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
            X = self.sigmoid(np.dot(X, weights[i].T))
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

    # ======== Activation Functions ========

    @staticmethod
    def sigmoid(z):
        """
        The activation function. Takes an input z (can be a scalar, vector or matrix) and outputs a value between
        zero and one based on the input.
        
        :param z:   The input value
        :return:    A value (or vector/matrix of values) between 0 and 1
        """
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return 1 / (1 + np.exp(-np.clip(z, -100, 100)))
        else:
            return NeuralNetwork.sigmoid(np.array(z))

    @staticmethod
    def sigmoid_gradient(z):
        """
        The activation function's derivative function. Takes an input z (can be a scalar, vector or matrix) and outputs
        the sigmoid function's gradient value for that input.
        
        :param z:   The input value
        :return:    The sigmoid function's gradient for that input z
        """
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return np.multiply(z, (1 - z))
        else:
            return NeuralNetwork.sigmoid_gradient(np.array(z))

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

    def __init__(self, units: int, has_bias=True):
        self.units = units
        self.has_bias = has_bias
        self.layer = np.zeros(units+(has_bias*1))

        # if this layer has a bias, we have to set the first item to 1
        if has_bias:
            self.layer[0] = 1

