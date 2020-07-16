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

    x_train, x_test, t_train, t_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

    #
    # preprocessing
    #
    std = StandardScaler()
    std.fit(x_train)

    x_train_std = std.transform(x_train)
    x_test_std = std.transform(x_test)

    #
    # linear regression
    #
    reg = linear_model.LinearRegression()
    reg.fit(x_train_std, t_train)

    t_pred = reg.predict(x_test_std)

    print("Linear Regression")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(t_test, t_pred)))
    print("R2 score: {:.2f}".format(r2_score(t_test, t_pred)))

    #
    # ridge regression
    #
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit(x_train_std, t_train)

    t_pred = reg.predict(x_test_std)

    print("\nRidge Regression")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(t_test, t_pred)))
    print("R2 score: {:.2f}".format(r2_score(t_test, t_pred)))

    #
    # lasso
    #
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(x_train_std, t_train)

    t_pred = reg.predict(x_test_std)

    print("\nLASSO")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(t_test, t_pred)))
    print("R2 score: {:.2f}".format(r2_score(t_test, t_pred)))

    #
    # elastic-net
    #
    reg = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
    reg.fit(x_train_std, t_train)

    t_pred = reg.predict(x_test_std)

    print("\nElastic-Net")
    print("Mean Squared Error: {:.2f}".format(mean_squared_error(t_test, t_pred)))
    print("R2 score: {:.2f}".format(r2_score(t_test, t_pred)))
    
