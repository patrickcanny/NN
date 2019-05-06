import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment=None

# Read Data and take a small sample
df = pd.read_csv('../data/tox21_dense_train.csv')
df.drop('Unnamed: 0', inplace=True, axis=1)

# Normalize data between 0 and 1
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

train = df.head(100)
test = df.tail(15)

# Split into labels and features
labels = train[799].values
feature_set = train.loc[:, train.columns != 799].values
labels = np.array([[x] for x in labels])
# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
# labels = np.array([[1,0,0,1,1]])
# labels = labels.reshape(5,1)

# Initialize weight
np.random.seed(42)
weights = np.random.rand(800,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

print("Training...")
for epoch in range(20000):
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step2
    z = sigmoid(XW)


    # backpropagation step 1
    error = z - labels

    # print(error.sum())
    if (epoch % 2000 == 0):
        print('.')

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

print("Done!")

def predict(train):
    true_values = train[799].values
    train.drop(799, inplace=True, axis=1)
    train = train.values
    preds = []
    for row in train:
        preds.append(sigmoid(np.dot(row, weights) + bias))
    print("R-Squared Accuracy: {}".format(r2_score(true_values, preds)))

predict(test)

# single_point = np.array([1,0,0])
# result = sigmoid(np.dot(single_point, weights) + bias)
# print(result)
