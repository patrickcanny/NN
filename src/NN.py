'''
EECS 738
Neural Network Implementation
Patrick Canny and Liam Ormiston
'''

class NN:
    def __init__(self, _input, y):
        self.input = _input
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weightsa = np.random.rand(4,1)
        self.y = y
        self.out = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

