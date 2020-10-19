import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def random_forest_classification():
    #
    # load dataset
    #
    breast = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    #
    # decision tree
    #
    clf = RandomForestClassifier(n_estimators=10, criterion="gini", max_features="auto", max_leaf_nodes=5,
                                 random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    #
    # performance evaluation
    #
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    f_score = 2 * precision * recall / (precision + recall)

    print("Accuracy {:.2f}%".format(accuracy * 100))
    print("Precision, Positive predictive value(PPV) {:.2f}%".format(precision * 100))
    print("Recall, Sensitivity, True positive rate(TPR) {:.2f}%".format(recall * 100))
    print("Specificity, True negative rate(TNR) {:.2f}%".format(specificity * 100))
    print("Negative predictive value(NPV) {:.2f}%".format(npv * 100))
    print("F-Score {:.2f}%".format(f_score * 100))


def random_forest_boundary():
    #
    # load dataset
    #
    iris = load_iris()

    plt.figure(figsize=(18, 12))
    for i, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        X = iris.data[:, pair]
        y = iris.target

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

        #
        # decision tree
        #
        clf = RandomForestClassifier(n_estimators=10, criterion="gini", max_features="auto", max_leaf_nodes=5,
                                     random_state=0)
        clf.fit(X, y)

        z = clf.predict(zz).reshape(xx.shape)

        #
        # visualization
        #
        plt.subplot(2, 3, i + 1)
        plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]), edgecolors="k", s=20)
        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    plt.show()


def random_forest_regression():
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
        # random forest
        #
        reg = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=0)
        reg.fit(X, y.ravel())

        y_pred = reg.predict(t)

        #
        # visualization
        #
        plt.plot(t, y_pred, color=color, label="max_depth=" + str(depth))

    plt.title("Random Forest Regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    random_forest_classification()

    random_forest_boundary()

    random_forest_regression()
    
