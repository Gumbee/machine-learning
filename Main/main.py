import numpy as np
import numpy.matlib
import Data.DataManager as DataManager
import time

from matplotlib import pyplot as plt
from NeuralNetwork.NeuralNetwork import NeuralNetwork as NeuralNetwork


def get_mean_correct(prediction, y):
    num_right = 0

    for i in range(0, len(prediction)):
        if np.array_equal(prediction[i], y[i]):
            num_right += 1

    print('Accuracy:', (num_right / (len(y) * 1.)) * 100)


def main():
    network = NeuralNetwork(400)
    network.add_hidden_layer(100)
    network.add_output_layer(10)

    X, y, X_val, y_val, X_test, y_test = DataManager.get_handwriting_data(0.8, 0.2)

    print('\nStarting timer...')
    t = time.time()
    # network.train(X, y, 850, alpha=1.5, reg_lambda=0.5)
    network.fmin(X, y, 1, 800)
    t = time.time()-t
    print("\nProcess finished in", '{:6.3f}'.format(t), 'seconds\n')

    print(network.predict(np.matlib.zeros((1, 400))))

    get_mean_correct(network.predict(X), y)
    get_mean_correct(network.predict(X_val), y_val)

    predictions = network.predict(X_val)

    confidence = network.feed_forward(X_val)

    network.cost_function(X, y)

    print("\nVisualizing....\n")

    plt.ion()
    gray = DataManager.read_image('../Data/DataFiles/number2.png')
    DataManager.display_image(gray)

    prediction = network.predict(np.matrix(np.ravel(gray.T)))
    conf = network.feed_forward(np.matrix(np.ravel(gray.T)))

    print("\nPrediciton for zwei: ", (int(np.where(prediction[0] == 1)[0]) + 1) % 10,
          " with confidence:", conf[2][0][(int(np.where(prediction[0] == 1)[0]))])

    input_var = input('\rPress Enter to continue or \'q\' to exit....')

    DataManager.visualize(X_val, y_val, predictions, confidence)
    DataManager.visualize(network.model['weights'][0][:, 1:])

main()
