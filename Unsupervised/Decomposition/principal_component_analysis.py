import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split

np.random.seed(0)

n_dim = 15


def principal_component_analysis():
    #
    # load dataset
    #
    breast = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    #
    # principal component analysis
    #
    pca = PCA(n_components=2)

    X_train_pca = pca.fit_transform(X_train)

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], breast.target_names):
        plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], color=c, label=l)
    plt.legend()
    plt.title("Principal Component Analysis")
    plt.xlabel("first component")
    plt.ylabel("second component")
    plt.show()

    #
    # explained variance ratio
    #
    pca = PCA(n_components=n_dim)
    pca.fit(X_train)

    plt.figure(figsize=(8, 6))
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.title("Explained Variance Ratio")
    plt.xlabel("n-th component")
    plt.ylabel("cumulative contribution ratio")
    plt.xticks([i for i in range(n_dim)], [i + 1 for i in range(15)])
    plt.show()


def kernel_pca():
    #
    # make dataset
    #
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

    #
    # PCA
    #
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    #
    # kernel PCA
    #
    kpca = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)

    #
    # visualization
    #
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], y):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=c, label=l)
    plt.title("Original")
    plt.subplot(1, 3, 2)
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], y):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=c, label=l)
    plt.title("PCA")
    plt.subplot(1, 3, 3)
    for c, i, l in zip(["#1f77b4", "#ff7f0e"], [0, 1], y):
        plt.scatter(X_kpca[y == i, 0], X_kpca[y == i, 1], color=c, label=l)
    plt.title("Kernel PCA")
    plt.show()


if __name__ == "__main__":
    principal_component_analysis()

    kernel_pca()
    
