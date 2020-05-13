import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression(theta_0=np.zeros(x_train.shape[1]))
    model.step_size = lr
    model.theta = model.fit(x_train, y_train)

    # predict the prob of samples in validation set
    exp_val = np.zeros((y_eval.shape[0], 1))
    for i in range(y_eval.shape[0]):
        exp_val[i] = model.predict(x_eval[i])

    np.savetxt(pred_path, exp_val)

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Logistic regression model parameters, including intercept.
        """
        # *** START CODE HERE ***
        m = y.shape[0]
        n = x.shape[1]
        theta = self.theta

        theta_old = theta + 1

        while np.linalg.norm(theta - theta_old, 1) > self.eps:
            summ = 0
            for i in range(m):
                x_i = np.transpose([x[i]])
                z = np.dot(theta, x_i)
                h_theta = np.exp(z)
                summ += (y[i] - h_theta) * x[i]
            theta_old = theta
            theta = theta + self.step_size * (1./ m) * summ

        return theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given logistic
        regression parameters theta. Input will not have an intercept term
        (i.e. not necessarily x[0] = 1), but theta expects an intercept term.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        # calculate the predicted y values
        x_t = np.transpose(x)
        exp_val = np.exp(np.dot(self.theta, x_t))
        return exp_val
        # *** END CODE HERE ***

