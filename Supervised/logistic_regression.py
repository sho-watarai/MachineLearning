import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    #
    # load dataset
    #
    breast = load_breast_cancer()

    x_train, x_test, t_train, t_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    #
    # preprocessing
    #
    std = StandardScaler()
    std.fit(x_train)

    x_train_std = std.transform(x_train)
    x_test_std = std.transform(x_test)

    #
    # logistic regression
    #
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=100)
    clf.fit(x_train_std, t_train)

    t_pred = clf.predict(x_test_std)

    #
    # performance evaluation
    #
    tn, fp, fn, tp = confusion_matrix(t_test, t_pred).ravel()

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
    plot_confusion_matrix(clf, x_test_std, t_test, cmap="hot")
    plot_precision_recall_curve(clf, x_test_std, t_test)
    plot_roc_curve(clf, x_test_std, t_test)

    plt.show()
    
