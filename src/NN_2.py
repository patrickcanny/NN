'''
Neural Network Implementation
Patrick Canny & Liam Ormiston
EECS 738
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Silence Errors
pd.options.mode.chained_assignment=None

# Read Data
df = pd.read_csv('../data/tox21_dense_train.csv')
df.drop('Unnamed: 0', inplace=True, axis=1)

# Normalize data between 0 and 1
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Split data into training and testing pieces
train = df.head(100)
test = df.tail(15)

# Split into labels and features
labels = train[799].values
feature_set = train.loc[:, train.columns != 799].values
labels = np.array([[x] for x in labels])

# Initialize weights
np.random.seed(42)
weights = np.random.rand(800,1)
bias = np.random.rand(1)
lr = 0.05

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Sigmoid Derivitave Function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Train Model
print("Training...")
for epoch in range(20000):
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    # feedforward step2
    # calculate sigmoid function
    z = sigmoid(XW)

    # backpropagation step 1
    # compute error
    error = z - labels

    # backpropagation step 2
    # look at error and modify stuff
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T

    # Modify weights
    weights -= lr * np.dot(inputs, z_delta)

    # Modify Bias
    for num in z_delta:
        bias -= lr * num

    #print a nice dot
    if (epoch % 2000 == 0):
        print('.')


print("Done!")

# Predict value for tox21 dataset unseen values
def predict(train):
    true_values = train[799].values
    train.drop(799, inplace=True, axis=1)
    train = train.values
    preds = []
    for row in train:
        preds.append(sigmoid(np.dot(row, weights) + bias))
    print("R-Squared Accuracy: {}".format(r2_score(true_values, preds)))

predict(test)
