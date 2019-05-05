'''
EECS 738
Neural Network Implementation
Patrick Canny and Liam Ormiston
'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd

# Sigmoid is our activation Function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NN:
    def __init__(self, _input, y):
        self.input = _input
        self.weights1 = np.random.rand(self.input.shape[1], self.input.shape[0])
        self.weights2 = np.random.rand(self.input.shape[0],1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    # X = np.array([[0,0,1],
    #               [0,1,1],
    #               [1,0,1],
    #               [1,1,1]])
    # y = np.array([[0],[1],[1],[0]])

    # nn = NN(X,y)

    # for i in range(1500):
    #     nn.feedforward()
    #     nn.backprop()

    # print(nn.output)
    # print(accuracy_score(y, nn.output))
    df = pd.read_csv('../data/tox21_dense_train.csv')
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df = df.head(10)
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    labels = df[799].values
    data = df.loc[:, df.columns != 799].values
    labels = np.array([[x] for x in labels])
    nn = NN(data, labels)
    for i in range(1500):
        nn.feedforward()
        nn.backprop()
        if (i % 100 == 0):
            print(nn.weights1)
            print(nn.weights2)
    print(nn.output)
