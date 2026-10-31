import numpy as np


class Regression:
    def __init__(self, input_size=784, output_size=10, bias=True, learning_rate=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.weights = None

    def fit(self, x,y):

        x_design = np.ones((x.shape[0], 1))
        x_design = np.column_stack((x_design, x))
        Xt = x_design.T
        Xt_y = Xt @ y
        self.weights = np.linalg.solve(Xt @ x_design, Xt_y)
        return self.weights

    def predict(self, x):
        x_design = np.ones((x.shape[0], 1))
        x_design = np.column_stack((x_design, x))
        return np.argmax(x_design @ self.weights)