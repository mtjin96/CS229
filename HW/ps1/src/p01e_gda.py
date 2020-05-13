import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(d): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    model = GDA(theta_0=np.zeros((x_train.shape[1], 1)))
    model.theta = model.fit(x_train, y_train)

    # predict the prob of samples in validation set
    p_x = np.zeros((y_val.shape[0], 1))
    for i in range(y_val.shape[0]):
        p_x[i] = model.predict(x_val[i])

    # plot the decision boundary
    #save_path = "output/p01e_gda_1.png"
    #util.plot(x_val, y_val, model.theta, save_path, correction=1.0)

    # save to pred_path
    np.savetxt(pred_path, p_x)

    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        # Find phi, mu_0, mu_1, and sigma
        m = x.shape[0]
        n = x.shape[1]

        # formulate the indicator
        I0, I1 = [], []
        for i in range(m):
            if y[i] == 0:
                I0.append(i)
            if y[i] == 1:
                I1.append(i)

        phi = len(I1)/m * 1.0

        mu0 = np.mean(x[I0], axis=0)
        mu1 = np.mean(x[I1], axis=0)

        sigma = np.zeros((n, n))

        for j in I0:
            sigma += np.dot((np.transpose([x[j] - mu0])), [(x[j] - mu0)])

        for j in I1:
            sigma += np.dot((np.transpose([x[j] - mu1])), [(x[j] - mu1)])

        sigma = (1.0 / m) * sigma

        # derive theta from other parameters
        theta1 = np.dot(np.linalg.inv(sigma), np.transpose([mu1-mu0]))
        d1 = np.dot(np.dot(mu0, np.linalg.inv(sigma)), np.transpose([mu0]))
        d2 = np.dot(np.dot(mu1, np.linalg.inv(sigma)), np.transpose([mu1]))
        theta0 = - np.log((1 - phi) / phi) + 0.5 * d1 - 0.5 * d2
        theta = np.zeros((1 + n, 1))
        theta[0] = theta0

        for i in range(len(theta1)):
            theta[i + 1] = theta1[i]

        return theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given linear coefficients theta.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        n = x.shape[0]
        theta_x = np.dot(x, self.theta[1:n+1])
        p_y1 = 1 / (1 + np.exp(-(theta_x + self.theta[0])))

        return p_y1

        # *** END CODE HERE
