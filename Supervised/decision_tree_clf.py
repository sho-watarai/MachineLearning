import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def decision_tree_classification():
    #
    # load dataset
    #
    breast = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    #
    # decision tree
    #
    clf = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=5, random_state=0)
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

    #
    # visualization
    #
    plot_confusion_matrix(clf, X_test, y_test, cmap="hot")
    plot_precision_recall_curve(clf, X_test, y_test)
    plot_roc_curve(clf, X_test, y_test)

    plt.figure(figsize=(8, 6))
    plot_tree(clf, filled=True, feature_names=breast.feature_names)

    #
    # decision tree structure
    #
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=int)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("\nThe binary tree structure has %s nodes and has the following tree structure:" % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to node %s." % (
                node_depth[i] * "\t", i, children_left[i], feature[i], threshold[i], children_right[i]))

    print()

    #
    # decision tree used to predict sample
    #
    node_indicator = clf.decision_path(X_test)
    leave_id = clf.apply(X_test)

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

    print("Rules used to predict sample %s: " % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)" % (node_id, sample_id, feature[node_id],
                                                                       X_test[sample_id, feature[node_id]],
                                                                       threshold_sign, threshold[node_id]))
    #
    # common node
    #
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids))
    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree" % (sample_ids, common_node_id))
    print("It is {:.2f}%% of all nodes.".format(100 * len(common_node_id) / n_nodes))

    plt.show()


def decision_tree_boundary():
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
        clf = DecisionTreeClassifier(criterion="gini", random_state=0)
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


if __name__ == "__main__":
    decision_tree_classification()

    decision_tree_boundary()
    
