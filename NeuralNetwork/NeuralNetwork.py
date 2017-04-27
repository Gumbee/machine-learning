import numpy as np
import scipy.optimize as opt


def ravel(weights):
    b = []
    for matrix in weights:
        b = np.hstack((b, np.array(matrix).ravel()))
    return b


def unravel(vector, layers):
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

    def add_hidden_layer(self, units, has_bias=True):
        # create new layer
        layer = NeuralLayer(units, has_bias)

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
        weights = unravel(theta, self.model['layers'])
        model = {'weights': weights, 'layers': self.model['layers']}

        feed_values = self.feed_forward(X, model)

        J = self.cost_function(X, y, reg_lambda, model, feed_values)
        gradients = ravel(self.gradient(X, y, reg_lambda, model, feed_values))

        return J, gradients

    def cost_function(self, X, y, reg_lambda=1., model=None, feed_values=None):
        """
        Calculates the cost of the current network model and returns the value

        :param X: input on which the network is/was being trained
        :param y: output corresponding to the input on which the network is/was being trained
        :param reg_lambda: regularization term for the weights of the network
        :param model: (optional) evaluate the cost function for a specific model. Default: Current Model
        :param feed_values: (optional) calculate the cost for the given feed_forward values
        :return:
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

        first_term = np.clip(output, 1e-88, 1)
        second_term = np.clip(np.subtract(1, output), 1e-88, 1)

        J = 0
        for j in range(0, layers[L-1].units):
            J += (-y[:, j]).T.dot(np.log(first_term[:, j])) - (np.subtract(1, y[:, j])).T.dot(np.log(second_term[:, j]))

        J *= (1./m)

        for weight in weights:
            square = np.square(weight[:, 1:])
            J += (reg_lambda/(2.*m))*(np.sum(square))

        return J

    def gradient(self, X, y, reg_lambda=1., model=None, feed_values=None):
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

    def fmin(self, X, y, reg_lambda=0.5, max_iter=600):
        # get optimized result from opt.fmin

        result = opt.fmin_tnc(func=self.cost_wrapper, x0=ravel(self.model['weights']), args=(X, y, reg_lambda), maxfun=max_iter)

        self.model['weights'] = unravel(result[0], self.model['layers'])

    def train(self, X, y, max_iter=5000, alpha=0.1, reg_lambda=1, print_progress=True, debug_mode=True):
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

    def predict(self, x, threshold=0, model=None):
        weights, layers = self.parse_model(model)

        # only get the last layer as it contains our output
        output = self.feed_forward(x)[len(layers)-1]

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

    def print_network(self):
        for layer in self.model['layers']:
            print(layer.layer)

    # ======== Activation Functions ========

    @staticmethod
    def sigmoid(z):
        if type(z).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return 1 / (1 + np.exp(-np.clip(z, -100, 100)))
        else:
            return NeuralNetwork.sigmoid()

    @staticmethod
    def sigmoid_gradient(a):
        if type(a).__module__ == np.__name__:
            # np.clip is used to prevent overflowing
            return np.multiply(a, (1-a))
        else:
            return NeuralNetwork.sigmoid_gradient()

    # ======== Verification Functions ========

    def check_gradients(self, X, y, gradients, reg_lambda, model=None, epsilon=1e-4):
        weights, layers = self.parse_model(model)

        for w in range(0, len(weights)):
            m, n = weights[w].shape
            for i in range(0, m):
                for j in range(0, n):
                    initial_weight = weights[w][i, j]
                    weights[w][i, j] = initial_weight + epsilon
                    plus = self.cost_function(X, y, reg_lambda, {'weights': weights, 'layers': layers})
                    weights[w][i, j] = initial_weight - epsilon
                    num_grad = (plus - self.cost_function(X, y, reg_lambda, {'weights': weights, 'layers': layers}))/(2*epsilon)
                    weights[w][i, j] = initial_weight
                    if gradients[w][i, j]-num_grad > 1e-9:
                        print('Numerical:', num_grad)
                        print('Algorithm:', gradients[w][i, j])
                        raise Exception('Gradients do not match!')

    # ======== Util Functions ========

    def get_model(self):
        return self.model

    def parse_model(self, model=None):
        if model is None:
            weights = self.model['weights']
            layers = self.model['layers']
        else:
            weights = model['weights']
            layers = model['layers']

        return weights, layers

    @staticmethod
    def fix_shape(X):
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
        print('\033[91m', '{:─>63s}'.format(''), '\033[0m')

    @staticmethod
    def print_table_entry(First, Second, Third, Fourth, Fifth):
        print('\033[91m', '{:>4d}'.format(First), '{:1s}'.format('|'), '{:>5d}'.format(Second), '{:>1s}'.format('|'),
              '{:>15.6e}'.format(Third), '{:>1s}'.format('|'), '{:>15.6e}'.format(Fourth), '{:>1s}'.format('|'),
              '{:>10.3f}'.format(Fifth), '{:>1s}'.format('|'), '\033[0m')


class NeuralLayer(object):
    def __init__(self, units, has_bias=True):
        self.units = units
        self.has_bias = has_bias
        self.layer = np.zeros(units+(has_bias*1))

        # if this layer has a bias, we have to set the first item to 1
        if has_bias:
            self.layer[0] = 1
