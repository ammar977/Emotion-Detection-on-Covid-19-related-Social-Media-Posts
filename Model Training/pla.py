import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
import matplotlib.lines as mlines

# from sklearn.metrics import accuracy_score
def visualize_scatter(df, feature1=0, feature2=1, target=2, weights=[1, 1, 1],
                      title=''):
    """
        Scatter plot feature1 vs feature2.
         +/- binary labels.
          - weights: [w1, w2, b]
    """

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[target]])
    ax = df.plot(x=feature1, y=feature2,kind='scatter', label=target, c=colors)
    ax.legend(target)
    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin,xmax )
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='crimson')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feature1), str(feature2))
    ax.set_title(title)

    plt.show()

def step_funct(y):
    return np.heaviside(y,0)

def weightInitialization(n_features):
    w = np.zeros((1, n_features))
    b = 0
    return w, b


def sigmoid_activation(result):
    final_result = 1 / (1 + np.exp(-result))
    return final_result

def perceptron(X, y, lr, epochs):
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n + 1, 1))

    # Empty list to store how many examples were
    # misclassified at every iteration.
    n_miss_list = []
    cost_list = []

    # Training.
    for epoch in range(epochs):

        # variable to store #misclassified.
        n_miss = 0
        cost = 0

        # looping for every example.
        for idx, x_i in enumerate(X):

            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # Calculating prediction/hypothesis.
            y_hat = sigmoid_activation(np.dot(x_i.T, theta))

            # Updating if the example is misclassified.

            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += lr * ((y[idx] - y_hat) * x_i)

                # Incrementing by 1.
                n_miss += 1
                cost = -1 * np.mean((y[idx] * np.log(y_hat)) +
                                    ((1 - y[idx]) * (np.log(1 - y_hat))))

        # Appending number of misclassified examples
        # at every iteration.
        n_miss_list.append(n_miss)
        cost_list.append(cost)


    return theta, n_miss_list, cost_list



def main(argv):
    #
    # Load the data set
    #
    datafile = pd.read_csv(argv[0], header=None, names=['feature1', 'feature2', 'y'])
    datafile['y'] = step_funct(datafile['y'])
    print(datafile, datafile.shape)
    outfile = argv[1]
    open(outfile,"w")

    # Get number of features
    n_features = datafile.shape[1]
    print('Number of Features', n_features)

    lr = 0.06
    epochs=8000
    theta, n_miss_list, cost_list = perceptron(np.array(datafile[['feature1','feature2']]), np.array(datafile['y']), lr, epochs)

    print(f'theta {theta}')
    print(f'n misses {n_miss_list}')
    print(f'cost list {cost_list}')
    visualize_scatter(datafile, feature1='feature1', feature2='feature2', target='y', weights=[theta[1],theta[2],theta[0]],
                      title='Try It')


if __name__ == "__main__":
    main(sys.argv[1:])