import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    #
    # load dataset
    #
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None)

    spam = df.values

    X_train, X_test, y_train, y_test = train_test_split(spam[:, :-1], spam[:, -1], test_size=0.2, random_state=0)

    #
    # preprocessing
    #
    norm = MinMaxScaler()

    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)

    #
    # naive bayes
    #
    clf = ComplementNB()
    clf.fit(X_train_norm, y_train)

    y_pred = clf.predict(X_test_norm)

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
    
