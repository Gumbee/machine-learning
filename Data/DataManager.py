import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def get_data(training_ratio=0.6, validation_ratio=0.2):
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


def display_image(img_vector, size=1):
    plt.figure(figsize=(size, size))
    plt.imshow(img_vector, interpolation='nearest', cmap='gray')
    plt.show()
    plt.pause(0.1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def read_image():
    return rgb2gray(mpimg.imread('../Data/DataFiles/number2.png'))


def visualize(X, y=None, predictions=None, confidence=None):
    idx = np.random.permutation(len(X))

    i = 0
    plt.ion()
    input_key = ''
    while not input_key == 'q' and i < len(X):
        img = np.array(X[idx[i]]).reshape((20, 20)).T
        display_image(img)

        if predictions is not None:
            print('\n{:<14s}'.format('Prediction:'), '{:^2d}'.format((int(np.where(predictions[idx[i]] == 1)[0]) + 1) % 10))
        if y is not None:
            print('{:<14s}'.format('Value:'), '{:^2d}'.format((int(np.where(y[idx[i]] == 1)[0]) + 1) % 10))
        if confidence is not None:
            print('Confidence:', confidence[2][idx[i]][(int(np.where(predictions[idx[i]] == 1)[0]))])

        input_key = input('\rPress Enter to continue or \'q\' to exit....')
        plt.close()
        i += 1

    plt.ioff()
