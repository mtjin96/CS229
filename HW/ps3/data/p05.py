import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt


def main():
    A = imread('peppers-large.tiff')
    plt.imshow(A)
    plt.show()

    im_small = imread('peppers-small.tiff')
    plt.imshow(im_small)
    plt.show()

    k = 16
    centroid = kmeans(im_small, k)

    # assign each example in the large image to the closest cluster using centroid
    dim = A.shape[0]
    A = np.reshape(A, (-1, 3))
    diffs = []
    for c in centroid:
        diff = np.linalg.norm(A - c, axis=1)
        diffs.append(diff)

    # Join the array "diff" along a new axis
    c_i = np.argmin(diffs, axis=0)

    # Compress the large image A
    compress_A = np.zeros((A.shape[0], A.shape[1]), dtype=int)
    for j in range(k):
        ind_j = np.where(c_i == j)
        compress_A[ind_j] = centroid[j]

    compress_A = compress_A.reshape(dim, dim, 3)
    plt.imshow((compress_A))
    plt.show()


def kmeans(A, k):
    # initialize centroid by randomly picking k training examples,
    # and set the cluster centroids to be equal to the values of these k examples
    A = np.reshape(A, (-1, 3))
    #A = A.astype(np.float32)
    m = A.shape[0]
    ind = np.random.choice(np.arange(m), size=k, replace=False)
    centroid = A[ind]

    iter = 0
    centroid = np.array(centroid)
    c_i = c_i_old = None
    while c_i_old is None or not np.array_equal(c_i, c_i_old):
        iter += 1
        c_i_old = c_i
        # Assigning each training example x_i to the closest cluster centroid miu_j
        diffs = []
        for c in centroid:
            diff = np.linalg.norm(A - c, axis=1)
            diffs.append(diff)

        c_i = np.argmin(diffs, axis=0)
        #print("c_i_old: ", c_i_old)
        #print("c_i: ", c_i)

        # Moving each cluster centroid miu_j to the mean of the points assigned to it
        miu_js = []
        for j in range(k):
            ind_j = np.where(c_i == j)
            miu_j = A[ind_j].mean(axis=0)
            miu_js.append(miu_j)

        centroid = np.array(miu_js)
        print("iteration: ", iter)
    return centroid


if __name__ == "__main__":
    main()

