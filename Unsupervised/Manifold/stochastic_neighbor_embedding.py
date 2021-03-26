import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

np.random.seed(0)


if __name__ == "__main__":
    #
    # load dataset
    #
    digits = load_digits(n_class=10)

    X = digits.data
    y = digits.target

    #
    # t-distribution Stochastic Neighbor Embedding
    #
    tsne = TSNE(n_components=2, perplexity=30, init="pca", n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    for i, l in zip(np.arange(10), y):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1])
    for i, l, w, h in zip(np.arange(10), y, X_tsne[:, 0], X_tsne[:, 1]):
        plt.annotate(i, xy=(w + 0.001, h + 0.001), xytext=(0, 0), textcoords="offset points", ha="right", va="bottom")
    plt.title("t-distribution Stochastic Neighbor Embedding")
    plt.show()

    #
    # perplexity
    #
    plt.figure(figsize=(30, 6))
    for j, ppl in enumerate([5, 10, 20, 30, 50]):
        tsne = TSNE(n_components=2, perplexity=ppl, init="pca", n_iter=1000)
        X_tsne = tsne.fit_transform(X)

        plt.subplot(1, 5, j + 1)
        for i, l in zip(np.arange(10), y):
            plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1])
        for i, l, w, h in zip(np.arange(10), y, X_tsne[:, 0], X_tsne[:, 1]):
            plt.annotate(i, xy=(w + 0.001, h + 0.001), xytext=(0, 0), textcoords="offset points", ha="right",
                         va="bottom")
        plt.title("perplexity=%d" % ppl)
    plt.show()
    
