import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)


if __name__ == "__main__":
    #
    # load dataset
    #
    iris = load_iris()

    X = iris.data[:, :2]
    y = iris.target

    #
    # k-nearest neighbor
    #
    clf = KNeighborsClassifier(n_neighbors=15, metric="minkowski")
    clf.fit(X, y)

    #
    # visualization
    #
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    z = clf.predict(zz).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["blue", "green", "red"]))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["b", "g", "r"]), edgecolor='k', s=20)  # training points
    plt.title("k-Nearest Neighbor Classification")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    
