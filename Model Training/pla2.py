import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy

from sklearn.metrics import accuracy_score

def weightInitialization(n_features):
    w = np.zeros((1, n_features))
    b = 0
    return w, b


def sigmoid_activation(result):
    final_result = 1 / (1 + np.exp(-result))
    return final_result


def model_optimize(w, b, X, Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid_activation(np.dot(w, X.T) + b)
    Y_T = Y.T
    cost = (-1 / m) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    #

    # Gradient calculation
    dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / m) * (np.sum(final_result - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    coeff_values=[]
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w, b, X, Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        if cost <= 0.00 :
            break
        #

        if (i % 4 == 0):
            coeff_values.append(np.column_stack((b,w)))
            costs.append(cost)
            print("Cost after %i iteration is %f" %(i, cost))

    # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs, coeff_values


def predict(final_pred, m):
    y_pred = np.zeros((1, m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

def main(argv):
    #
    # Load the data set
    #
    datafile = pd.read_csv(argv[0], header=None, names=['feature1', 'feature2', 'y'])
    print(datafile, datafile.shape)
    outfile = argv[1]
    open(outfile,"w")
    df = np.array(datafile)
    X, y = df[:, 0:2:], df[:, 2:3:]
    X_train, y_train = X[0:11,0:2], y[0:11]
    X_test, y_test = X[10:,0:2], y[10:]

    # Get number of features
    n_features = X_train.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization(n_features)
    # Gradient Descent
    coeff, gradient, costs, coeff_values = model_predict(w, b, X_train, y_train, learning_rate=0.0001, no_iterations=80)
    # Final prediction
    w = coeff["w"]
    # print(w[:,1])
    b = coeff["b"]
    weights_list = np.column_stack((w,b)).reshape((3,))
    print(f'wl {weights_list}')
    print('Optimized weights', w)
    print('Optimized intercept', b)
    #
    final_train_pred = sigmoid_activation(np.dot(w, X_train.T) + b)
    final_test_pred = sigmoid_activation(np.dot(w, X_test.T) + b)
    #
    m_tr = X_train.shape[0]
    m_ts = X_test.shape[0]
    #
    # used this just to check accuracy
    # y_tr_pred = predict(final_train_pred, m_tr)
    # print('Training Accuracy', accuracy_score(y_tr_pred.T, y_train))
    # #
    # y_ts_pred = predict(final_test_pred, m_ts)
    # print('Test Accuracy', accuracy_score(y_ts_pred.T, y_test))
    coeff_values=np.array(coeff_values).reshape((len(coeff_values),3))

    pd.DataFrame(coeff_values).to_csv(outfile, header=None, index=False)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])