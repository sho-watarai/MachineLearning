import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

np.random.seed(0)


def svc_linear():
    #
    # load dataset
    #
    breast = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    #
    # preprocessing
    #
    std = StandardScaler()

    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    #
    # linear SVC
    #
    clf = LinearSVC()
    clf.fit(X_train_std, y_train)

    y_pred = clf.predict(X_test_std)

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

    #
    # visualization
    #
    plot_confusion_matrix(clf, X_test_std, y_test, cmap="hot")
    plot_precision_recall_curve(clf, X_test_std, y_test)
    plot_roc_curve(clf, X_test_std, y_test)

    plt.show()


def svc_margin():
    #
    # make dataset
    #
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    plt.figure(figsize=(18, 6))
    for i, C in enumerate([1, 0.05, 100]):
        #
        # linear SVC
        #
        clf = SVC(C=C, kernel="linear")
        clf.fit(X, y)

        z = clf.decision_function(zz).reshape(xx.shape)

        #
        # visualization
        #
        plt.subplot(1, 3, i + 1)
        plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]), edgecolors="k", s=20)
        plt.contour(xx, yy, z, colors="k", levels=[-1, 0, 1], alpha=0.5,
                    linestyles=["--", "-", "--"])  # decision boundary and margin
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidths=1, facecolors="none",
                    edgecolors="k")  # support vector
        plt.title("C={}".format(C))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    plt.show()


def svc_kernel():
    #
    # make dataset
    #
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    plt.figure(figsize=(16, 12))
    for i, kernel in enumerate(["linear", "poly", "rbf", "sigmoid"]):
        #
        # kernel SVC
        #
        clf = SVC(C=1.0, kernel=kernel, degree=2, gamma=2, coef0=1.0)
        clf.fit(X, y)

        z = clf.decision_function(zz).reshape(xx.shape)

        #
        # visualization
        #
        plt.subplot(2, 2, i + 1)
        plt.pcolormesh(xx, yy, z > 0, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]))
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]), edgecolors="k")
        plt.contour(xx, yy, z, colors="k", levels=[-1, 0, 1], alpha=0.5,
                    linestyles=["--", "-", "--"])  # decision boundary and margin
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors="none", zorder=10,
                    edgecolors="k")  # support vector
        plt.title(kernel)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.show()


def svc_gamma():
    #
    # make dataset
    #
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    plt.figure(figsize=(18, 6))
    for i, gamma in enumerate([1, 0.1, 10]):
        #
        # linear SVC
        #
        clf = SVC(kernel="rbf", gamma=gamma)
        clf.fit(X, y)

        z = clf.decision_function(zz).reshape(xx.shape)

        #
        # visualization
        #
        plt.subplot(1, 3, i + 1)
        plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#ff7f0e"]), edgecolors="k", s=20)
        plt.contour(xx, yy, z, colors="k", levels=[-1, 0, 1], alpha=0.5,
                    linestyles=["--", "-", "--"])  # decision boundary and margin
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidths=1, facecolors="none",
                    edgecolors="k")  # support vector
        plt.title("C={}".format(gamma))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    plt.show()


def svc_multiclass():
    #
    # load dataset
    #
    iris = load_iris()

    X = iris.data[:, :2]
    y = iris.target

    #
    # multi-class SVC
    #
    clf = SVC(C=1.0, kernel="linear", decision_function_shape="ovr", break_ties=True)
    clf.fit(X, y)

    #
    # visualization
    #
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    z = clf.predict(zz).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e"]), edgecolor="k", s=20)
    plt.title("Multi-Class Support Vector Classification")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    svc_linear()

    svc_margin()

    svc_kernel()

    svc_gamma()

    svc_multiclass()
    
