import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import sklearn.datasets
from numpy import ndarray
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils import shuffle

path = os.getcwd()

folds = 4 + 1
k_max = 11


def kNN_classifier():
    X_y = load_iris()
    (X, y) = shuffle(X_y["data"], X_y["target"], random_state=0)

    '''
    Normalize data
    '''
    for c in range(0, len(X[0])):

        x = X[:, c]

        min_v = x.min()
        max_v = x.max()

        for i in range(0, len(x)):
            X[:, c][i] = (X[:, c][i] - min_v) / (max_v - min_v)

    rows = len(X)
    columns = len(X[0])
    labels = np.unique(y)

    segment_len = int(rows / folds)

    for k in range(3, k_max + 1, 2):
        print("k: ", k)

        for fold in range(0, folds - 1):
            validation_start = fold * segment_len

            hits = 0

            for validation in range(validation_start, validation_start + segment_len):
                v = X[validation]

                distances = list()
                for training in chain(range(0, validation_start),
                                      range(validation_start + segment_len, segment_len * (folds - 1))):
                    t = X[training]

                    distance = 0
                    for i in range(0, columns):
                        distance += math.pow(v[i] - t[i], 2)
                    distance = math.sqrt(distance)

                    distances.append((distance, training))

                distances.sort()

                nearestN = distances[:k]
                results = [0] * labels

                for n in nearestN:
                    results[y[n[1]]] += 1

                winner = 0
                for r in range(0, len(results)):
                    if results[r] > results[winner]:
                        winner = r

                if winner == y[validation]:
                    hits += 1

            print("fold ", fold, " accuracy: ", hits / segment_len)

        print()


def kNN_regressor():
    X_y = load_iris()
    (X, y) = shuffle(X_y["data"], X_y["target"], random_state=0)

    '''
    Normalize data
    '''

    columns = len(X[0])
    rows = len(X[:, 0])

    for c in range(0, columns):

        max_v = X[:, c].max()
        min_v = X[:, c].min()

        for i in range(0, rows):
            X[:, c][i] = (X[:, c][i] - min_v) / (max_v - min_v)

    segment_len = int(len(X[:, 0]) / folds)

    for k in range(3, k_max + 1, 2):
        print("k: ", k)

        for fold in range(0, folds - 1):
            validation_start = fold * segment_len

            mea = 0
            y_true = list()
            y_pred = list()

            for validation in range(validation_start, validation_start + segment_len):
                v = X[validation]

                distances = list()
                for training in chain(range(0, validation_start),
                                      range(validation_start + segment_len, segment_len * (folds - 1))):
                    t = X[training]

                    distance = 0
                    for i in range(0, columns):
                        distance += math.pow(v[i] - t[i], 2)
                    distance = math.sqrt(distance)

                    distances.append((distance, training))

                distances.sort()

                nearestN = distances[:k]

                total_distance = 0.0
                for d in nearestN:
                    total_distance += d[0]

                Y = 0.0
                for n in nearestN:
                    Y += y[n[1]] * (n[0] / total_distance)

                mea += Y - y[validation]

                y_pred.append(Y)
                y_true.append(y[n[1]])

            error = mean_absolute_percentage_error(y_true, y_pred)
            print("fold ", fold, " error: ", error)


def kNN_SciKit_classifier():
    X_y = load_iris()
    (X, y) = shuffle(X_y["data"], X_y["target"], random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    print("score", neigh.score(X_test, y_test))


def kNN_SciKit_regressor():
    X_y = load_boston()
    (X, y) = shuffle(X_y["data"], X_y["target"], random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X_train, y_train)
    Y = neigh.predict(X_test)

    error = mean_absolute_percentage_error(Y, y_test)
    print("error", error)


if __name__ == '__main__':
    kNN_classifier()
    kNN_regressor()

    print()

    kNN_SciKit_classifier()
    kNN_SciKit_regressor()
