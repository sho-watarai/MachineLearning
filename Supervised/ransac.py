import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.datasets import make_regression

n_samples = 1000
n_outlier = 50


if __name__ == "__main__":
    #
    # make dataset
    #
    X, y, coef = make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10,
                                 coef=True, random_state=0)

    #
    # add outlier
    #
    X[:n_outlier] = 3 + 0.5 * np.random.normal(size=(n_outlier, 1))
    y[:n_outlier] = -3 + 10 * np.random.normal(size=n_outlier)

    x = np.arange(X.min(), X.max())[:, np.newaxis]

    #
    # linear regression
    #
    linear = linear_model.LinearRegression()
    linear.fit(X, y)

    y_linear = linear.predict(x)

    #
    # random sample consensus
    #
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=100, min_samples=50,
                                          residual_threshold=None, random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    y_ransac = ransac.predict(x)

    print("Estimated Coefficients")
    print("True: {:.2f}, LinearRegression: {:.2f}, RANSAC: {:.2f}".format(
        coef, linear.coef_[0], ransac.estimator_.coef_[0]))

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    plt.scatter(X[inlier_mask], y[inlier_mask], marker=".", label="inlier")
    plt.scatter(X[outlier_mask], y[outlier_mask], marker="x", label="outlier")
    plt.plot(x, y_linear, color="#2ca02c", label="LinearRegression")
    plt.plot(x, y_ransac, color="gold", label="RANSAC")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
