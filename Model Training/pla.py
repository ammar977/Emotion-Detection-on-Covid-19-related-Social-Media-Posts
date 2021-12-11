import copy
import math

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np


#
# Perceptron implementation
#
class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.coef_values = []
        self.globalerr =0

    '''
    Stochastic Gradient Descent

    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.coef_values = []
        for i in range(self.n_iterations):
            self.globalerr = 0
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1
                self.globalerr += (expected_value - predicted_value)**2

            if (i % 4 == 0):
                self.coef_values.append(copy.deepcopy(self.coef_))
                print(f"Iteration {i} : {self.globalerr} , {len(X)} RMSE = { math.sqrt(self.globalerr/len(X))}")

        print(f"\nDecision boundary (line) equation: {self.coef_[0]}*x + {self.coef_[1]}*y + {self.coef_[2]} = 0\n")


    '''
    Net Input is sum of weighted input signals
    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''
    Activation function is fed the net input and the unit step function
    is executed to determine the output.
    '''

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)

    '''
    Prediction is made on the basis of output of activation function
    '''

    def predict(self, X):
        return self.activation_function(X)

    '''
    Model score is calculated based on comparison of
    expected value and predicted value
    '''

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if (target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_



def main(argv):
    #
    # Load the data set
    #
    datafile = pd.read_csv(argv[0], header=None, names=["feature1", "feature2", "y"])
    print(datafile, datafile.shape)
    outfile = argv[1]
    open(outfile,"w")
    df = np.array(datafile)
    X, y = df[:, 0:2:], df[:, 2:3:]
    X_train, y_train = X[0:11,0:2], y[0:11]
    X_test, y_test = X[10:,0:2], y[10:]

    # Instantiate CustomPerceptron
    #
    prcptrn = CustomPerceptron(n_iterations=80, random_state=1, learning_rate=0.00001)
    #
    # Fit the model
    #
    prcptrn.fit(X_train, y_train)
    print(type(prcptrn.coef_))
    pd.DataFrame(prcptrn.coef_values).to_csv(outfile, index=False, header=None)
    print(prcptrn.coef_, np.array(prcptrn.coef_values))
    #
    # Score the model
    #
    print(prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train))


if __name__ == "__main__":
    main(sys.argv[1:])