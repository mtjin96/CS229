import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    m = y_val.shape[0]
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Plot data
    # Run on the test set, and use np.savetxt to save outputs to pred_path

    # find MSE using different tau values
    MSE = np.zeros((len(tau_values), 1))
    mse_min = 0.5
    tau_min = 0.5

    for i in range(len(tau_values)):
        model = LocallyWeightedLinearRegression(tau=tau_values[i])
        model.fit(x_train, y_train)
        # predict the y values in validation set
        y_pred = np.zeros([y_val.shape[0], 1])
        for j in range(m):
            y_pred[j] = model.predict(x_val[j])

        #plt.plot(x_val, y_val, 'bx', x_val, y_pred, 'ro')
        #plt.show()

        # calculate the MSE
        mse = 0
        for k in range(m):
            diff = y_val[k] - y_pred[k]
            mse += (1.0 / m) * diff * diff
        MSE[i] = mse

        # find the tau that achieves the smallest MSE
        if MSE[i] < mse_min:
            tau_min = model.tau
            mse_min = MSE[i]

        # plot data
        plt.plot(x_train[:, 1], y_train, 'bx', x_val[:, 1], y_pred, 'ro')
        #plt.show()


    # use the tau that min the MSE to predict test data
    model_t = LocallyWeightedLinearRegression(tau=tau_min)
    model_t.fit(x_train, y_train)

    # predict the y values in validation set
    m_test = y_test.shape[0]
    y_test_pred = np.zeros([m, 1])
    for j in range(m_test):
        y_test_pred[j] = model_t.predict(x_test[j])

    # save prediction
    np.savetxt(pred_path, y_test_pred)

    # calculate MSE using the best tau
    mse_test = 0
    for i in range(m_test):
        diff_t = y_test[i] - y_test_pred[i]
        mse_test += (1.0 / m_test) * diff_t * diff_t
    #print("tau: ", tau_min)
    #print("MSE: ", mse_test)

    # *** END CODE HERE ***
