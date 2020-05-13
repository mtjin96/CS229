import matplotlib.pyplot as plt
import numpy as np
import os


PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # dimension
    m, n = x.shape
    x_data = x.copy()
    np.random.shuffle(x_data)
    k_group = np.split(x_data, K)
    mu1 = np.mean(k_group[0], 0)
    mu2 = np.mean(k_group[1], 0)
    mu3 = np.mean(k_group[2], 0)
    mu4 = np.mean(k_group[3], 0)
    mu = np.asmatrix(np.array([mu1, mu2, mu3, mu4]))
    cov1 = np.cov(k_group[0].T)
    cov2 = np.cov(k_group[1].T)
    cov3 = np.cov(k_group[2].T)
    cov4 = np.cov(k_group[3].T)
    sigma = np.array([cov1, cov2, cov3, cov4])
    #print("mu: ", mu)
    #print(mu.shape)
    #print("sigma: ", sigma)
    #mu = np.asmatrix(np.random.random((K, n)))
    #sigma = np.array([np.asmatrix(np.identity(n)) for i in range(K)])
    #w = np.asmatrix(np.empty((m, K), dtype=float))

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.array([np.ones(K) / K] * m)

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        it += 1
        # (1) E-step: Update your estimates in w
        m, n = x.shape
        for i in range(m):
            deno = 0
            P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-n / 2.) \
                              * np.exp(-.5 * np.dot(np.dot((x[i, :] - mu), np.linalg.inv(s)), (x[i, :] - mu).T))
            for j in range(K):
                num = P(mu[j], sigma[j]) * phi[j]
                #num = sp.multivariate_normal.pdf(x[i, :], mu[j].A1, sigma[j]) * phi[j]
                deno += num
                w[i, j] = num
            w[i, :] = w[i, :] / deno

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            summ = w[:, j].sum() #np.sum(w[:, j])
            phi[j] = (1.0 / m) * summ

            mu_j = np.zeros(n)
            sigma_j = np.zeros((n, n))

            for i in range(m):
                mu_j += (x[i, :] * w[i, j]) #w[i, j] * x[i, :] / summ
                sigma_j += w[i, j] * ((x[i, :] - mu[j, :]).T * (x[i, :] - mu[j, :])) #w[i, j] * np.dot(np.transpose(x[i, :] - mu[j, :]), (x[i, :] - mu[j, :])) / summ

            mu[j] = mu_j / summ
            sigma[j] = sigma_j / summ

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        print("prev: ", prev_ll)
        ll = 0
        for i in range(m):
            P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-n / 2.) \
                              * np.exp(-.5 * np.dot(np.dot((x[i, :] - mu), np.linalg.inv(s)), (x[i, :] - mu).T))
            sum_z = 0
            for j in range(K):
                sum_z += P(mu[j], sigma[j]) * phi[j]
                #sum_z += sp.multivariate_normal.pdf(x[i, :], mu[j].A1, sigma[j]) * phi[j]
            ll += np.log(sum_z)
        print("UnS ll: ", ll)
        # *** END CODE HERE ***
    print("iterations: ", it)
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        it += 1
        # (1) E-step: Update your estimates in w
        m, n = x.shape
        m_tilde, n_tilde = x_tilde.shape
        for i in range(m):
            deno = 0
            P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-n / 2.) \
                              * np.exp(-.5 * np.dot(np.dot((x[i, :] - mu), np.linalg.inv(s)), (x[i, :] - mu).T))
            for j in range(K):
                num = P(mu[j], sigma[j]) * phi[j]
                # num = sp.multivariate_normal.pdf(x[i, :], mu[j].A1, sigma[j]) * phi[j]
                deno += num
                w[i, j] = num
            w[i, :] = w[i, :] / deno

        # (2) M-step: Update the model parameters phi, mu, and sigma
        unique, z_counts = np.unique(z, return_counts=True)
        for j in range(K):
            summ = w[:, j].sum() #np.sum(w[:, j])
            phi[j] = (summ + alpha * z_counts[j]) / (m + alpha * m_tilde)

            mu_j = np.zeros(n)
            sigma_j = np.zeros((n, n))

            for i in range(m):
                mu_j += (x[i, :] * w[i, j]) #w[i, j] * x[i, :] / summ
                sigma_j += w[i, j] * ((x[i, :] - mu[j, :]).T * (x[i, :] - mu[j, :]))
                #w[i, j] * np.dot(np.transpose(x[i, :] - mu[j, :]), (x[i, :] - mu[j, :])) / summ

            for i in range(m_tilde):
                if z[i] == j:
                    indicator = 1
                else:
                    indicator = 0
                mu_j += alpha * indicator * x_tilde[i, :]
                sigma_j += alpha * indicator * ((x_tilde[i, :] - mu[j, :]).T * (x_tilde[i, :] - mu[j, :]))

            mu[j] = mu_j / (summ + alpha * z_counts[j])
            sigma[j] = sigma_j / (summ + alpha * z_counts[j])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        print("prev: ", prev_ll)
        ll = 0
        # unlabeled part
        for i in range(m):
            P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-n / 2.) \
                              * np.exp(-.5 * np.dot(np.dot((x[i, :] - mu), np.linalg.inv(s)), (x[i, :] - mu).T))
            sum_z = 0
            for j in range(K):
                sum_z += P(mu[j], sigma[j]) * phi[j]
                # sum_z += sp.multivariate_normal.pdf(x[i, :], mu[j].A1, sigma[j]) * phi[j]
            ll += np.log(sum_z)

        # labeled part
        #unique, z_counts = np.unique(z, return_counts=True)
        for i in range(m_tilde):
            P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-n / 2.) \
                              * np.exp(-.5 * np.dot(np.dot((x_tilde[i, :] - mu), np.linalg.inv(s)), (x_tilde[i, :] - mu).T))

            prob = P(mu[j], sigma[j]) * phi[int(z[i][0])]
            #for j in range(K):
            #    if z[i] == j:
            #        indicator = 1
            #    else:
            #        indicator = 0
            #    sum_z2 += P(mu[j], sigma[j]) * phi[j] #(z_counts[j] / m_tilde)
            ll += alpha * np.log(prob)
        print("SSL ll: ", ll)

        # *** END CODE HERE ***
    print("iterations: ", it)
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
