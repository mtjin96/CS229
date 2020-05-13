import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # Plot data
    # No need to use np.savetxt in this problem

    m = y_val.shape[0]
    model = LocallyWeightedLinearRegression(tau=tau)

    # fit training data
    model.fit(x_train, y_train)

    # predict the y values in validation set
    y_pred = np.zeros([y_val.shape[0], 1])
    for i in range(m):
        y_pred[i] = model.predict(x_val[i])

    # calculate the MSE
    mse = 0
    for i in range(m):
        diff = y_val[i] - y_pred[i]
        mse += (1.0 / m) * diff * diff
    #print("MSE: ", mse)

    # plot data
    plt.plot(x_train[:, 1], y_train, 'bx', x_val[:, 1], y_pred, 'ro')
    #plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set."""
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        # *** START CODE HERE ***
        m_train = self.x.shape[0]
        w_i = [0] * m_train

        # construct the weight matrix by comparing each sample in training with the given one in validation
        for i in range(m_train):
            x_i = self.x[i]
            num = np.square(np.linalg.norm((x_i - x), 2))
            deno = 2 * self.tau * self.tau
            w_i[i] = np.exp(-num / deno)

        W = np.diag(np.diag(np.diag(w_i)))
        r = np.dot(np.dot(np.transpose(self.x), W), self.x)
        z = np.dot(np.dot(np.transpose(self.x), W), self.y)
        theta = np.dot(np.linalg.inv(r), z)

        y_i = np.dot(x, theta)
        return y_i
        # *** END CODE HERE ***
