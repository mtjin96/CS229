import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

def main():
    im_large = imread('peppers-large.tiff')
    plt.imshow(im_large)
    plt.show()

    im_small = imread('peppers-small.tiff')
    plt.imshow(im_small)
    plt.show()

    X = im_small.reshape(-1, 3)
    num_clusters = 16
    centroids, _, err_history = kmean(X, 16)
    print(centroids)

    dists_list = []
    dim = im_large.shape[0]

    X_large = im_large.reshape(-1, 3)
    for c in centroids:
        ds = np.sqrt(np.sum((X_large - c) ** 2, axis=1))
        dists_list.append(ds)
    assign = np.stack(dists_list).argmin(axis=0)

    compressed = np.zeros_like(X_large)
    for k in range(num_clusters):
        idxes = np.where(assign == k)[0]
        compressed[idxes] = centroids[k]
    compressed = compressed.reshape(dim, dim, 3)
    plt.imshow((compressed * 255).astype(np.uint8))
    plt.show()

def kmean(X, num_clusters=16):
    X = X.astype(np.float32)

    # pikcing random data points from the data as the initial centroids to aovid empty cluster
    _idxes = np.random.choice(np.arange(X.shape[0]), size=num_clusters, replace=False)
    centroids = X[_idxes]

    err_history = []
    err = 1e6
    iter = 0
    while err > 1:
        iter += 1
        dists_list = []
        for c in centroids:
            ds = np.sqrt(np.sum((X - c) ** 2, axis=1))
            dists_list.append(ds)

        assign = np.stack(dists_list).argmin(axis=0)

        # new centroids
        nc_list = []
        for k in range(num_clusters):
            idxes = X[np.where(assign == k)[0]]
            nc_list.append(X[np.where(assign == k)[0]].mean(axis=0))

        nc = np.stack(nc_list)
        err = np.sum(np.abs(nc - centroids))
        err_history.append(err)    
        centroids = nc
        print(iter)
    return centroids, assign, err_history


if __name__ == "__main__":
    main()
