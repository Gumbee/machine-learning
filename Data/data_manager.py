import numpy as np
import pickle as pickle
import matplotlib.image as mpimg


def get_handwriting_data(training_ratio=0.6, validation_ratio=0.2):
    """
    Reads the handwriting data, splits it up into training set, cross validation set and test set and returns the result.
    
    :param training_ratio:      The fraction of the data that should be used as the training set
    :param validation_ratio:    The fraction of the data that should be used as the cross validation set (the remaining
                                fraction is used as test set).
    :return:                    X -> Training set, y -> Training set's output, X_val -> Validation set, y_val -> Validation set's output,
                                X_test -> Test set, y_test -> Test set's output
    """
    input_data = []
    output_data = []

    with open('../Data/DataFiles/handwritingData.txt') as data:
        for line in data:
            current_line = line.split(',')
            input_data.append(current_line)

    with open('../Data/DataFiles/handwritingDataResult.txt') as data:
        for line in data:
            number_vec = np.zeros(10)
            number_vec[int(line) - 1] = 1
            output_data.append(number_vec)

    input_data = np.array(input_data).astype(np.float)
    output_data = np.array(output_data)

    m = len(input_data)
    # get a random permutation of all the indices so we don't just train on certain digits
    idx = np.random.permutation(len(input_data))

    training_end = int(training_ratio*m)
    validation_end = int(training_end + validation_ratio*m)

    X = input_data[idx[0:training_end], :]
    y = output_data[idx[0:training_end], :]

    X_val = input_data[idx[training_end:validation_end], :]
    y_val = output_data[idx[training_end:validation_end], :]

    X_test = input_data[idx[validation_end:], :]
    y_test = output_data[idx[validation_end:], :]

    return X, y, X_val, y_val, X_test, y_test


def make_output(x):
    """
    Helper function to create the output vectors for the MNIST data set
    
    :param x:   The digit that should be converted to a output vector
    :return:    The output vector
    """

    output = np.zeros(10)
    output[x-1] = 1

    return output


def get_mnist_data():
    """
    Reads the MNIST data set into the variables X, y, X_val, y_val, X_test, y_test and returns them
    
    :return:    X -> Training set, y -> Training set's output, X_val -> Validation set, y_val -> Validation set's output,
                X_test -> Test set, y_test -> Test set's output
    """
    # Data can be downloaded here: http://deeplearning.net/data/mnist/mnist.pkl.gz

    file = open('../Data/DataFiles/mnist.pkl', 'rb')
    X, X_val, X_test = pickle.load(file, encoding='latin1')
    file.close()

    y = np.array([make_output(x) for x in X[1]])
    X = X[0]

    y_val = np.array([make_output(x) for x in X_val[1]])
    X_val = X_val[0]

    y_test = np.array([make_output(x) for x in X_test[1]])
    X_test = X_test[0]

    return X, y, X_val, y_val, X_test, y_test


def rgb2gray(img_matrix: np.matrix):
    """
    Converts a matrix of rgb values to grayscale values.
    
    :param img_matrix:  The image matrix that is to be converted to grayscale values
    :return:            The matrix as grayscale values
    """
    return np.dot(img_matrix[..., :3], [0.299, 0.587, 0.114])


def read_image(path: str):
    """
    Converts an image to a matrix of grayscale values.
    
    :return: The image's grayscale matrix
    """
    return rgb2gray(mpimg.imread(path))


def generate_data(m: int = 100, noise: float = 2, degree: int = 2):
    """
    Generates data consisting of a function f(x) = ax+bx^2+cx^3... of the desired polynomial degree. If specified, noise
    will be added to the data.
    
    :param m:       The number of examples (size of the data)
    :param noise:   Number specifying how much noise should be added (the higher the number, the more noise)
    :param degree:  The degree of the function that generates the data
    :return:        The data's input and output set
    """
    thetas = np.matrix(np.random.normal(np.zeros(degree), 0.1))

    x = create_polynomial_features(m, degree)

    y = np.random.normal(x.dot(thetas.T), noise)

    return x, y


def create_polynomial_features(m: int, degree: int = 2):
    """
    Creates a data input set with polynomial features up to a specified degree.
    
    :param m:       The number of examples (size of the data)
    :param degree:  The max degree that should be generated
    :return:        The data with polynomial features
    """
    X_out = np.ones((m, degree))

    for i in range(0, m):
        for j in range(1, degree+1):
            X_out[i, j-1] = (i-m/2)**j

    return X_out


