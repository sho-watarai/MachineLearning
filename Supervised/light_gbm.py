import lightgbm as lgb
import os

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

is_write = False


if __name__ == "__main__":
    #
    # load dataset
    #
    breast = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=0)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(breast.feature_names))
    test_data = lgb.Dataset(X_test, label=y_test)
    param = {"num_leaves": 21, "objective": "binary", "metric": "auc"}

    num_round = 10

    model = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    if is_write:
        model.save_model("./lightgbm.txt")
    if os.path.isfile("./lightgbm.txt"):
        model = lgb.Booster(model_file="./model.txt")

    lgb.plot_tree(model)
    lgb.plot_importance(model)

    #
    # performance evaluation
    #
    y_pred = model.predict(X_test) > 0.5

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
    
