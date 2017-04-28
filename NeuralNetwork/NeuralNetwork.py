import numpy as np
import scipy.optimize as opt


def ravel(weights):
    """
    Takes a list of weight matrices and converts them into one big vector.
    
    :param weights: The list of weights matrices
    :return: A vector containing all elements of the weight matrices
    """
    b = []
    for matrix in weights:
        b = np.hstack((b, np.array(matrix).ravel()))
    return b


def unravel(vector, layers):
    """
    Takes a vector and reconstructs the weight matrices based on the given layer structure.
    
    :param vector: The vector containing all weight elements
    :param layers: The list of layers from which the weight matrices dimensions are reconstructed
    :return: The weights as list of matrices
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

    def add_hidden_layer(self, units):
        """
        Adds a new hidden layer to the network. If there exists an output already, the new hidden layer is inserted
        between the current last hidden layer and the output.
        
        :param units: The number of hidden units contained in this hidden layer
        :return: None
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
            self.model['weights'][len(self.model['weights'])-1] = np.random.rand(output_layer.units, units + (has_bias*1)) * (2*self.EPSILON) - self.EPSILON

        print('Added a new layer with ', units, ' units.')

    def add_output_layer(self, units):
        """
        Adds an output layer to the network. If a output layer exists already, that layer will be replaced with the new
        one. Any new hidden layers added to the network after an output layer was added will be inserted between the
        last hidden layer and the output layer.
        
        :param units: The number of output units contained in this output layer
        :return: None
        """
        # create new layer
        layer = NeuralLayer(units, False)
        # get previous layer
        prev_layer = self.model['layers'][len(self.model['layers']) - 1]
        # create random weights matrix
        weight = np.random.rand(units, prev_layer.units + (prev_layer.has_bias * 1)) * (2 * self.EPSILON) - self.EPSILON

        if not self.has_output:
            self.has_output = True
            self.model['layers'].append(layer)
            self.model['weights'].append(weight)

            print('Added a new output layer with ', units, ' units.')
        else:
            # override existing values
            self.model['layers'][len(self.model['layers'])-1] = layer
            self.model['weights'][len(self.model['weights'])-1] = weight

            print('Output layer overridden with ', units, ' units.')

    # ======== Gradient Descent Functions ========

    def cost_wrapper(self, theta, X, y, reg_lambda=1.):
        """
        Wrapper function for scipy's optimization functions.
        
        :param theta: One dimensional vector containing all weight elements of the network
        :param X: The training set on which the network is trained
        :param y: The training set's output
        :param reg_lambda: (optional) Regularization term for the weights. Default: 1
        :return: cost, gradients
        """

        weights = unravel(theta, self.model['layers'])
        model = {'weights': weights, 'layers': self.model['layers']}

        feed_values = self.feed_forward(X, model)

        J = self.cost_function(X, y, reg_lambda, model, feed_values)
        gradients = ravel(self.gradient(X, y, reg_lambda, model, feed_values))

        return J, gradients

    def cost_function(self, X, y, reg_lambda=1., model=None, feed_values=None):
        """
        Evaluates the cost of the given model (standard network model if no model is passed as parameter) based on a 
        given training set X with output y. Optionally feed forward values can be passed on so that this method doesn't
        feed forward the training set again.
        
        :param X: input on which the model is/was being trained
        :param y: output corresponding to the input on which the model is/was being trained
        :param reg_lambda: (optional) regularization term for the weights of the network. Default: 1
        :param model: (optional) evaluate the cost function for a specific model. Default: Current model
        :param feed_values: (optional) calculate the cost for the given feed_forward values
        :return: cost of the model
        """

        weights, layers = self.parse_model(model)

        m, n, X = self.fix_shape(X)
        _, _, y = self.fix_shape(y)

        L = len(layers)
        W = len(weights)

        if feed_values is None:
            output = self.feed_forward(X, model)[L-1]
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

    def gradient(self, X, y, reg_lambda=1., model=None, feed_values=None):
        """
        Computes the gradients of the weight matrices of the provided model (standard network model if None is passed).
        
        :param X: The training set with which the network is trained
        :param y: The training set's output
        :param reg_lambda: (optional) Regularization term for the weights. Default: 1
        :param model: (optional) compute the gradients for a specific model. Default: Current model
        :param feed_values: (optional) calculate the cost for the given feed_forward values
        :return: gradients of the weights
        """
        weights, layers = self.parse_model(model)

        m, n, X = self.fix_shape(X)
        _, _, y = self.fix_shape(y)

        W = len(weights)
        L = len(layers)

        # check if the feed forward values are already provided or if we have to calculate them
        if feed_values is None:
            activations = self.feed_forward(X, model)
        else:
            activations = feed_values

        # allocate space
        deltas = [None]*L
        gradients = []

        # calculate deltas (layer errors)
        deltas[L-1] = (activations[L-1] - y).T

        for i in range(L-2, 0, -1):
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

    def fmin(self, X, y, reg_lambda=1, max_iter=600):
        """
        Trains the network with scipy's opt function with the given training set and the corresponding output and
        applies the trained model to the network.
        
        :param X: The training set with which the network is trained
        :param y: The training set's output
        :param reg_lambda: (optional) Regularization term for the weights. Default: 1
        :param max_iter: (optional) Maximal number of iterations before the function should end the training
        :return: None
        """
        # get optimized result from opt.fmin
        result = opt.fmin_tnc(func=self.cost_wrapper, x0=ravel(self.model['weights']), args=(X, y, reg_lambda), maxfun=max_iter)

        self.model['weights'] = unravel(result[0], self.model['layers'])

    def train(self, X, y, max_iter=5000, alpha=0.1, reg_lambda=1, debug_mode=True):
        """
        Trains the network with gradient descent with the given training set and the corresponding output and
        applies the trained model to the network.
        
        :param X: The training set with which the network is trained
        :param y: The training set's output
        :param max_iter: (optional) Maximal number of iterations before the function should end the training
        :param alpha: (optional) Learning Rate of the gradient descent algorithm. Default: 0.1
        :param reg_lambda: (optional) Regularization term for the weights. Default: 1
        :param debug_mode: (optional) True if debug mode should be turned on (outputs a table with important values)
        :return: None
        """
        if not type(X).__module__ == np.__name__ or not type(y).__module__ == np.__name__:
            self.train(np.array(X), np.array(y), max_iter, alpha, reg_lambda, print_progress, debug_mode)
            return

        print('\nTraining Neural Network...')

        initial_error = self.cost_function(X, y, reg_lambda)

        if debug_mode:
            self.print_table_header('P', 'IT', 'COST', 'CHNG', 'ASCL')
            self.print_table_entry(0, 0, initial_error, initial_error, 1.00)

        # keeps track of how many entries we've already printed
        entry_num = 1
        # factor by which the learning rate alpha is scaled
        alpha_scale = 1.
        # keep track of the previous iteration's error so we can calculate the relative change
        prev_cst = initial_error

        for t in range(0, max_iter):
            _, _, X = self.fix_shape(X)

            # calculate gradients
            gradients = self.gradient(X, y, reg_lambda)

            # update weights with gradients
            for i in range(0, len(self.model['weights'])):
                self.model['weights'][i] -= alpha*np.log10(t+1) * gradients[i]

            # reevaluate cost function
            cost = self.cost_function(X, y, reg_lambda)
            # get relative change of the cost function
            rel_chng = cost - prev_cst
            # update previous cost to current cost
            prev_cst = cost

            if debug_mode and t % 7 == 0:
                self.print_table_entry(entry_num, t+1, cost, rel_chng, alpha_scale)
                entry_num += 1

        print('\033[91m', '\n{:<15s}'.format('Initial Error:'), '{:5.6e}'.format(initial_error), '\n{:<15s}'.format('New Error:'), '{:>5.6e}'.format(self.cost_function(X, y, reg_lambda)), '\033[0m')

    # ======== Helper Functions ========

    def feed_forward(self, X, model=None):
        """
        Takes an input set X and a model (standard network model if no model is passed), feeds the input forward and
        returns the activation values for every layer.
        
        :param X: The input set which is fed forward
        :param model: (optional) compute the feed forward values for a specific model. Default: Current model
        :return: The activation values for every unit in every layer
        """
        if not type(X).__module__ == np.__name__:
            return self.feed_forward(np.array(X))

        m, n, X = self.fix_shape(X)

        thetas, layers = self.parse_model(model)

        activations = [X]

        for i in range(0, len(layers)-1):
            X = np.hstack((np.ones((m, 1)), X))
            X = self.sigmoid(np.dot(X, thetas[i].T))
            activations.append(X)

        return activations

    def predict(self, X, threshold=0, model=None):
        """
        Takes an input set and predicts the output based on the model (standard network model if nothing else is
        specified). If a threshold is specified then every output unit with a value greater than the threshold
        will output 1 and all else 0. If no threshold is specified then the output unit with the greatest value
        will output 1 and all other units will output 0.
        
        :param X: The input set
        :param threshold: (optional) The threshold which determines if a output unit outputs a 1 or 0. If the 
                          threshold is 0 then only the output unit with the greatest value will output 1.
        :param model: (optional) compute the output for a specific model. Default: Current model
        :return: 
        """
        weights, layers = self.parse_model(model)

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
        
        :param z: The input value
        :return: A value (or vector/matrix of values) between 0 and 1
        """
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return 1 / (1 + np.exp(-np.clip(z, -100, 100)))
        else:
            return NeuralNetwork.sigmoid()

    @staticmethod
    def sigmoid_gradient(z):
        """
        The activation function's derivative function. Takes an input z (can be a scalar, vector or matrix) and outputs
        the sigmoid function's gradient value for that input.
        
        :param z: The input value
        :return: The sigmoid function's gradient for that input z
        """
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return np.multiply(z, (1 - z))
        else:
            return NeuralNetwork.sigmoid_gradient()

    # ======== Verification Functions ========

    def check_gradients(self, X, y, gradients, reg_lambda, model=None, epsilon=1e-4, threshold=1e-9):
        """
        Numerically calculate the gradients based on a model (standard network model if no model is specified) and
        compare them to the given gradients. If they don't match, raise an error.
        
        :param X: The training set on which the model was trained
        :param y: The output corresponding to the training set
        :param gradients: The gradients which are to be checked
        :param reg_lambda: The regularization term used to train the mode
        :param model: (optional) The model which is to be used. Default: Current model
        :param epsilon: (optional) How accurate the numerical gradient should be (the smaller the better, but beware too small values)
        :param threshold: (optional) If the difference between the numerical gradient and the provided gradient is
                          bigger than the threshold an error will be raised
        :return: None
        """
        weights, layers = self.parse_model(model)

        for w in range(0, len(weights)):
            m, n = weights[w].shape
            # loop through all gradients
            for i in range(0, m):
                for j in range(0, n):
                    # store the initial weight
                    initial_weight = weights[w][i, j]
                    # add a small value to the initial weight
                    weights[w][i, j] = initial_weight + epsilon
                    # calculate the new cost function with the small value added to the weight element
                    plus = self.cost_function(X, y, reg_lambda, {'weights': weights, 'layers': layers})
                    # subtract a small value from the inital weight
                    weights[w][i, j] = initial_weight - epsilon
                    # calculate the new cost function with the small value subtracted to the weight element and save
                    # the difference between the cost where we added a value and the cost where we subtracted it
                    num_grad = (plus - self.cost_function(X, y, reg_lambda, {'weights': weights, 'layers': layers}))/(2*epsilon)
                    # restore the weight element's initial weight
                    weights[w][i, j] = initial_weight
                    if gradients[w][i, j]-num_grad > threshold:
                        print('Numerical:', num_grad)
                        print('Algorithm:', gradients[w][i, j])
                        # raise an error if the difference between the numerical gradient and the provided gradient
                        # is exceeding the threshold
                        raise Exception('Gradients do not match!')

    # ======== Util Functions ========

    def get_model(self):
        return self.model

    def parse_model(self, model=None):
        """
        Takes a model as argument (if no model is passed to the method then the network's current model will be used)
        and returns that model's weights and layers
        
        :param model: (optional) The model whose weights and layers should be returned. Default: Current model
        :return: The model's weights and layers
        """
        if model is None:
            weights = self.model['weights']
            layers = self.model['layers']
        else:
            weights = model['weights']
            layers = model['layers']

        return weights, layers

    @staticmethod
    def fix_shape(X):
        """
        Takes a scalar, vector or a matrix and returns it as a matrix.
        
        :param X: The scalar/vector/matrix
        :return: The output matrix's dimensions and the input as a matrix
        """
        if X.ndim > 1:
            m, n = np.shape(X)
        else:
            X = X.reshape((1, len(X)))
            m, n = np.shape(X)

        return m, n, X

    @staticmethod
    def print_table_header(First, Second, Third, Fourth, Fifth):
        print('\n\033[91m', '{:>4s}'.format(str(First)), '{:>1s}'.format('|'), '{:>5s}'.format(str(Second)), '{:>1s}'.format('|'),
              '{:>15s}'.format(str(Third)), '{:>1s}'.format('|'), '{:>15s}'.format(str(Fourth)), '{:>1s}'.format('|'),
              '{:>10s}'.format(str(Fifth)), '{:>1s}'.format('|'), '\033[0m')
        print()
        print('\033[91m', '{:─>63s}'.format(''), '\033[0m')

    @staticmethod
    def print_table_entry(First, Second, Third, Fourth, Fifth):
        print('\033[91m', '{:>4d}'.format(First), '{:1s}'.format('|'), '{:>5d}'.format(Second), '{:>1s}'.format('|'),
              '{:>15.6e}'.format(Third), '{:>1s}'.format('|'), '{:>15.6e}'.format(Fourth), '{:>1s}'.format('|'),
              '{:>10.3f}'.format(Fifth), '{:>1s}'.format('|'), '\033[0m')


class NeuralLayer(object):
    """
    Class that contains necessary information for a neural layer (e.g how many units).
    """

    def __init__(self, units, has_bias=True):
        self.units = units
        self.has_bias = has_bias
        self.layer = np.zeros(units+(has_bias*1))

        # if this layer has a bias, we have to set the first item to 1
        if has_bias:
            self.layer[0] = 1

