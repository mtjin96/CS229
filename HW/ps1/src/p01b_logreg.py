import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path

    model = LogisticRegression(theta_0=np.zeros((x_train.shape[1], 1)))
    model.theta = model.fit(x_train, y_train)

    # predict the prob of samples in validation set
    p_x = np.zeros([y_eval.shape[0], 1])
    for i in range(y_eval.shape[0]):
        p_x[i] = model.predict(x_eval[i])


    # plot decision boundary
    #save_path = "output/p01e_logreg_2.png"
    #util.plot(x_eval, y_eval, model.theta, save_path, correction=1.0)

    # save prediction
    np.savetxt(pred_path, p_x)

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

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
        #l_theta = 0
        l_grad = np.zeros((n, 1))
        #print(l_grad)
        l_hessian = np.zeros((n, n))

        # Newton's method to update theta

        # as long as not achieved max iteration number and eps, keep iterating to find theta
        iter_num = 0
        theta_old = theta + 1
        while iter_num < self.max_iter and np.linalg.norm(theta - theta_old, 1) > self.eps:
            l_grad = 0
            l_hessian = 0
            # calculate the gradient and hessian of log likelihood fn
            for i in range(m):
                x_i = np.transpose([x[i]])
                theta_x = np.dot(np.transpose(theta), x_i)
                h_theta = 1.0 / (1 + np.exp(-theta_x))
                l_grad += (1.0 / m) * (h_theta - y[i]) * x_i
                l_hessian += (1.0 / m) * h_theta * (1 - h_theta) * np.dot(x_i, [x[i]])
            iter_num += 1
            theta_old = theta
            theta = theta - np.dot(np.linalg.inv(l_hessian), l_grad)

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

        # calculate the predicted prob for input x
        x_i = np.transpose([x])
        p_x = 1 / (1 + np.exp(-np.dot(np.transpose(self.theta), x_i)))
        return p_x

        # *** END CODE HERE ***
