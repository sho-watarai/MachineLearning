import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(0)


if __name__ == "__main__":
    #
    # load dataset
    #
    iris = load_iris()

    X = iris.data
    y = iris.target
    names = iris.target_names

    #
    # Fisher Linear Discriminant Analysis
    #
    lda = LinearDiscriminantAnalysis(n_components=2)

    X2 = lda.fit_transform(X, y)

    #
    # visualization
    #
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    plt.figure(figsize=(8, 6))
    for c, i, l in zip(colors, [0, 1, 2], names):
        plt.scatter(X2[y == i, 0], X2[y == i, 1], color=c, label=l)
    plt.legend()
    plt.title("Fisher Linear Discriminant Analysis")
    plt.show()
    
