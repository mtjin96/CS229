import util
import matplotlib.pyplot as plt
import numpy as np


def plot_points(x, y):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y == -1, :]
    x_two = x[y == 1, :]

    plt.scatter(x_one[:, 1], x_one[:, 2], marker='x', color='red')
    plt.scatter(x_two[:, 1], x_two[:, 2], marker='o', color='blue')


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {-1, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model"""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        # plot decision boundary for the ith iteration listed in i_lst
        i_lst = [1, 2, 3, 10, 100, 200, 500, 1000, 10000, 30370, 40000, 50000]
        if i in i_lst:
            save_path = "output/p01_b_a" + str(i) + ".png"
            plot(X, Y, theta, save_path)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    plot_points(Xa, Ya)
    plt.savefig('output/p01_b_a.png')

    plot_points(Xb, Yb)
    plt.savefig('output/p01_b_b.png')

    #logistic_regression(Xa, Ya)
    logistic_regression(Xa, Ya)


if __name__ == "__main__":
    main()