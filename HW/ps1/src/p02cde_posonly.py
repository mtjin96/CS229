import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')


    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # load dataset
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model = LogisticRegression(theta_0=np.zeros((x_train.shape[1], 1)))
    model.theta = model.fit(x_train, t_train)

    # predict the labels in test set using only t-labels
    l = np.zeros((t_test.shape[0], 1))
    for i in range(t_test.shape[0]):
        l[i] = model.predict(x_test[i])
    np.savetxt(pred_path_c, l)


    # Part (d): Train on y-labels and test on true labels
    # load dataset
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_2 = LogisticRegression(theta_0=np.zeros([x_train.shape[1], 1]))
    model_2.theta = model_2.fit(x_train, y_train)

    # predict the labels in test set only using y-labels
    l_2 = np.zeros([y_test.shape[0], 1])
    for i in range(y_test.shape[0]):
        l_2[i] = model_2.predict(x_test[i])
    np.savetxt(pred_path_d, l_2)

    # Part (e): Apply correction factor using validation set and test on true labels
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    I1 = []
    for i in range(y_val.shape[0]):
        if y_val[i] == 1:
            I1.append(i)
    # calculate sample size of y_i = 1
    v_plus_sz = len(I1)

    # predict the labels in test set only using y-labels
    l_3 = np.zeros([y_val.shape[0], 1])
    for i in range(y_val.shape[0]):
        l_3[i] = model_2.predict(x_val[i])
    summ = sum(l_3[I1])

    # calculate alpha
    alpha = 1. / v_plus_sz * summ

    np.savetxt(pred_path_e, l_3)

    # Plot and use np.savetxt to save outputs to pred_path
    #save_path_c = "output/p02c.png"
    #util.plot(x_test, t_test, model.theta, save_path_c, correction=1.0)

    #save_path_d = "output/p02d.png"
    #util.plot(x_test, y_test, model_2.theta, save_path_d, correction=1.0)

    #save_path_e = "output/p02e.png"
    #util.plot(x_test, y_test, model_2.theta, save_path_e, correction=alpha)
#
    # *** END CODER HERE
