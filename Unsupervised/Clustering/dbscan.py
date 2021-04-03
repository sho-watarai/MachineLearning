import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons


if __name__ == "__main__":
    #
    # make dataset
    #
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    # k-means
    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)

    # hierarchical
    ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
    y_ac = ac.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
    y_dbscan = dbscan.fit_predict(X)

    #
    # visualization
    #
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1])

    plt.subplot(2, 2, 2)
    plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0, 1], c="#1f77b4", marker="o", s=40)
    plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1, 1], c="#ff7f0e", marker="s", s=40)
    plt.title("DBSCAN")

    plt.subplot(2, 2, 3)
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c="#1f77b4", marker="o", s=40)
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c="#ff7f0e", marker="s", s=40)
    plt.title("K-means")

    plt.subplot(2, 2, 4)
    plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c="#1f77b4", marker="o", s=40)
    plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c="#ff7f0e", marker="s", s=40)
    plt.title("Agglomerative")

    plt.show()
    
