import math

from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def mul_lin_reg(scoring_type):
    (X, y) = load_boston(return_X_y=True)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    remaining_features = list(range(0, len(X_train[0])))
    picked_features = list()

    best_score = -1

    while len(remaining_features) > 0:

        best_feature = -1

        for i in range(0, len(remaining_features)):
            feature = remaining_features[i]

            temp_picked = picked_features.copy()
            temp_picked.append(feature)

            reg = LinearRegression()
            reg.fit(X_train[:, temp_picked], y_train)

            score = 0
            if scoring_type == 0:
                score = 1-reg.score(X_test[:, temp_picked], y_test)

            if scoring_type == 1:
                rss = sum((reg.predict(X_test[:, temp_picked]) - y_test) ** 2)
                score = math.sqrt(rss/(len(X_test)-len(temp_picked)-1))

            if best_score < 0:
                best_score = score
            else:
                if best_score > score:
                    best_feature = feature
                    best_score = score

        if best_feature >= 0:
            remaining_features.remove(best_feature)
            picked_features.append(best_feature)
        else:
            break

    print(picked_features)
    reg = LinearRegression()
    reg.fit(X_train[:, picked_features], y_train)
    print(reg.score(X_test[:, picked_features], y_test))


def backw_reg():
    (X, y) = load_boston(return_X_y=True)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    remaining_features = list(range(0, len(X_train[0])))

    while True:
        X2 = sm.add_constant(X_train[:, remaining_features])
        est = sm.OLS(y_train, X2)
        est2 = est.fit()

        highest_pv = 0
        index = 0
        pvals = est2.pvalues

        for i in range(0, len(pvals)):
            pv = pvals[i]
            if pv > highest_pv:
                highest_pv = pv
                index = i

        if highest_pv < 0.2:
            break
        remaining_features.pop(index)

    print(remaining_features)
    reg = LinearRegression()
    reg.fit(X_train[:, remaining_features], y_train)
    print(reg.score(X_test[:, remaining_features], y_test))

if __name__ == "__main__":
    mul_lin_reg(0)
    mul_lin_reg(1)
    backw_reg()
