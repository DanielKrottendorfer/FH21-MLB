import math

import numpy as np
import scipy.stats
from sklearn.datasets import load_iris, load_boston, load_wine, load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, norm

import matplotlib.pyplot as plt


def main():
    (X, y) = load_diabetes(return_X_y=True)
    print(X)
    X = X[:, np.newaxis, 2]
    X_train1, X_test, y_train1, y_test = train_test_split(X, # the features
                                                      y, # the labels/target values
                                                      test_size=0.2, # size of the test set set aside, usually 0.2 or 0.33
                                                      shuffle=True, # if the dataset should be shuffled before splitting
                                                      random_state=99)

    reg = LinearRegression().fit(X_train1, y_train1)
    predictions = reg.predict(X_test)

    print("skl=> r2 score: ", reg.score(X_test, y_test), " MSE: ", mean_squared_error(predictions, y_test))

    rss = np.sum((predictions - y_test) ** 2)  # residual sum of square
    tss = np.sum((np.mean(y_test) - y_test) ** 2)  # total sum of squares
    r2_manual = 1 - (rss / tss)

    print("diy=> r2 score: ", r2_manual, " MSE: ", rss/len(y_test))

    print("p_vals:", ttest_ind(a=X_train1, b=X_test)[1])

    print("------------",len(X_test), len(y_test))

    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, predictions, color='blue', linewidth=3)
    plt.show()

    plt.hist(x=y_test, density=True, facecolor='g', alpha=0.75)
    mu = np.sum(y_test)/len(y_test)
    variance = np.var(y_test)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma))
    plt.show()


if __name__ == '__main__':
    main()
