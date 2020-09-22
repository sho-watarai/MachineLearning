import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

np.random.seed(0)


def knn_classification(k):
    #
    # load dataset
    #
    iris = load_iris()

    X = iris.data[:, :2]
    y = iris.target

    #
    # k-nearest neighbor
    #
    clf = KNeighborsClassifier(n_neighbors=k, metric="minkowski")
    clf.fit(X, y)

    #
    # visualization
    #
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    z = clf.predict(zz).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]), edgecolor="k", s=20)
    plt.title("k-Nearest Neighbor Classification")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def knn_regression(k):
    #
    # make dataset
    #
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X)
    y[::5] += (0.5 - np.random.rand(8, 1))

    t = np.linspace(0, 5, 400)[:, np.newaxis]

    #
    # k-nearest neighbor
    #
    reg = KNeighborsRegressor(n_neighbors=k, weights="distance")
    reg.fit(X, y)

    y_pred = reg.predict(t)

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, c="#ff7f0e", label="data")
    plt.plot(t, y_pred, c="#1f77b4", label="predict")
    plt.title("k-Nearest Neighbor Regression")
    plt.legend()
    plt.show()


def knn_unsupervised(k):
    #
    # load dataset
    #
    iris = load_iris()

    X = iris.data[:, :2]

    #
    # k-nearest neighbor
    #
    knn = NearestNeighbors(n_neighbors=k, metric="minkowski")
    knn.fit(X)

    distance, indices = knn.kneighbors(X)
    graph = knn.kneighbors_graph(X).toarray()

    print(distance.shape)
    print(indices.shape)

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    plt.imshow(graph, cmap="gray")
    plt.colorbar()
    plt.title("k-Nearest Neighbor Unsupervised")
    plt.show()


if __name__ == "__main__":
    knn_classification(k=15)

    knn_regression(k=5)

    knn_unsupervised(k=15)
    
