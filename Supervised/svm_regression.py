import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR

np.random.seed(0)


if __name__ == "__main__":
    #
    # make dataset
    #
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X)
    y[::5] += (0.5 - np.random.rand(8, 1))

    t = np.linspace(0, 5, 400)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="k", label="data", facecolor="none")
    for i, (kernel, color) in enumerate(zip(["linear", "rbf", "poly"], ["#1f77b4", "#2ca02c", "#ff7f0e"])):
        #
        # kernel svr
        #
        reg = SVR(C=100, kernel=kernel, degree=3, gamma=0.1, epsilon=0.1)
        reg.fit(X, y.ravel())

        y_pred = reg.predict(t)

        #
        # visualization
        #
        plt.plot(t, y_pred, color=color, label=kernel)

    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()
    
