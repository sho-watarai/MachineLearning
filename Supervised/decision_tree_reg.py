import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor


def decision_tree_regression():
    #
    # make dataset
    #
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X)
    y[::5] += (0.5 - np.random.rand(8, 1))

    t = np.linspace(0, 5, 400)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="k", label="data", facecolor="none")
    for i, (depth, color) in enumerate(zip([2, 5, 8], ["#1f77b4", "#2ca02c", "#ff7f0e"])):
        #
        # decision tree
        #
        reg = DecisionTreeRegressor(max_depth=depth)
        reg.fit(X, y)

        y_pred = reg.predict(t)

        #
        # visualization
        #
        plt.plot(t, y_pred, color=color, label="max_depth=" + str(depth))

    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


def decision_tree_regression_multi():
    #
    # make dataset
    #
    X = np.sort(200 * np.random.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - np.random.rand(20, 2))

    t = np.arange(-100, 100, 0.01)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.scatter(y[:, 0], y[:, 1], color="k", label="data", facecolor="none")
    for i, (depth, color) in enumerate(zip([2, 5, 8], ["#1f77b4", "#2ca02c", "#ff7f0e"])):
        #
        # decision tree
        #
        reg = DecisionTreeRegressor(max_depth=depth, random_state=0)
        reg.fit(X, y)

        y_pred = reg.predict(t)

        #
        # visualization
        #
        plt.scatter(y_pred[:, 0], y_pred[:, 1], color=color, label="max_depth=" + str(depth))

    plt.title("Multi-Output Decision Tree Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    decision_tree_regression()

    decision_tree_regression_multi()
    
