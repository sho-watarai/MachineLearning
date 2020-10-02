import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

np.random.seed(0)


def neural_network_curve():
    #
    # make dataset
    #
    x = np.linspace(-1, 1, 100)

    y1 = np.cos(np.pi * (x - 1))
    y2 = np.cos(np.pi * (x - 1)) - 1.5

    y = np.hstack((np.vstack((x, y1)), np.vstack((x, y2)))).T
    t = np.hstack((np.ones((100,)), np.zeros((100,))))

    xx, yy = np.meshgrid(x, np.linspace(np.floor(y.min()), np.floor(y.max()), 100))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    #
    # neural network: 1 hidden layer, tanh activation
    #
    clf = MLPClassifier(solver="lbfgs", alpha=1e-4, hidden_layer_sizes=(3, 2), activation="tanh")
    clf.fit(y, t)

    zz = clf.predict(zz).reshape(xx.shape)

    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(121)
    ax.plot(x, y1, x, y2)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.contourf(xx, yy, zz, cmap=ListedColormap(["#ff7f0e", "#1f77b4"]), alpha=0.8)

    #
    # feature space
    #
    weights = clf.coefs_
    biases = clf.intercepts_
    z = np.tanh(np.dot(y, weights[0]) + biases[0])

    ax = fig.add_subplot(122, projection="3d")
    ax.plot3D(z[:100, 0], z[:100, 1], z[:100, 2])
    ax.plot3D(z[100:, 0], z[100:, 1], z[100:, 2])

    plt.show()


def neural_network_circle():
    #
    # make dataset
    #
    X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05)

    #
    # neural network: 3 hidden layer, tanh activation
    #
    clf = MLPClassifier(solver="lbfgs", alpha=1e-4, hidden_layer_sizes=(10, 15, 10, 3), activation="tanh")
    clf.fit(X, y)

    weights = clf.coefs_
    biases = clf.intercepts_
    z1 = np.tanh(np.dot(X, weights[0]) + biases[0])
    z2 = np.tanh(np.dot(z1, weights[1]) + biases[1])
    z3 = np.tanh(np.dot(z2, weights[2]) + biases[2])
    z = np.tanh(np.dot(z3, weights[3]) + biases[3])

    #
    # feature space
    #
    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(121)
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], y):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=c, label=l)

    ax = fig.add_subplot(122, projection="3d")
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], y):
        ax.scatter3D(z[y == i, 0], z[y == i, 1], z[y == i, 2], color=c, label=l)

    plt.show()


if __name__ == "__main__":
    neural_network_curve()

    neural_network_circle()
    
