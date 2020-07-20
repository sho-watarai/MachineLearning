from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    #
    # load dataset
    #
    diabetes = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

    #
    # preprocessing
    #
    std = StandardScaler()
    
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

    #
    # linear regression
    #
    reg = linear_model.LinearRegression()
    reg.fit(X_train_std, y_train)

    y_pred = reg.predict(X_test_std)

    print("Linear Regression")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("R2 score: {:.2f}".format(r2_score(y_test, y_pred)))

    #
    # ridge regression
    #
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit(X_train_std, y_train)

    y_pred = reg.predict(X_test_std)

    print("\nRidge Regression")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("R2 score: {:.2f}".format(r2_score(y_test, y_pred)))

    #
    # lasso
    #
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X_train_std, y_train)

    y_pred = reg.predict(X_test_std)

    print("\nLASSO")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("R2 score: {:.2f}".format(r2_score(y_test, y_pred)))

    #
    # elastic-net
    #
    reg = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
    reg.fit(X_train_std, y_train)

    y_pred = reg.predict(X_test_std)

    print("\nElastic-Net")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("R2 score: {:.2f}".format(r2_score(y_test, y_pred)))
    
