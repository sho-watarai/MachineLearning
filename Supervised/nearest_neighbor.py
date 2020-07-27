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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]), edgecolor="k", s=20)
    ax.set_title("k-Nearest Neighbor Classification")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.show()
    
