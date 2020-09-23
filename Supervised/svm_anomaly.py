import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.svm import OneClassSVM

np.random.seed(0)


if __name__ == "__main__":
    #
    # make dataset
    #
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.vstack((X + 2, X - 2))
    X = 0.3 * np.random.randn(50, 2)
    X_test = np.vstack((X + 2, X - 2))
    X_outlier = np.random.uniform(low=-4, high=4, size=(25, 2))

    #
    # one-class SVM
    #
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outlier = clf.predict(X_outlier)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outlier[y_pred_outlier == 1].size

    #
    # visualization
    #
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    z = clf.predict(zz).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, cmap=ListedColormap(["white", "#1f77b4"]))
    plt.scatter(X_train[:, 0], X_train[:, 1], c="#1f77b4", s=40, edgecolors="k", label="train")  # training point
    plt.scatter(X_test[:, 0], X_test[:, 1], c="#2ca02c", s=40, edgecolors="k", label="test")  # test point
    plt.scatter(X_outlier[:, 0], X_outlier[:, 1], c="#ff7f0e", s=40, edgecolors="k", label="outlier")  # outlier point
    plt.legend()
    plt.title("One-Class Support Vector Machine")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()
    
