import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils import shuffle

path = os.getcwd()

folds = 4 + 1
k_max = 11


def kNN_classifier():
    df = pd.read_csv(os.path.join(path, "iris.csv"), sep=";")

    '''
    Normalize data
    '''
    for c in df.columns.values:
        if c == "class":
            continue

        min_v = df[c].min()
        max_v = df[c].max()

        for i in range(0, len(df[c].values)):
            df[c].iloc[i] = (df[c].iloc[i] - min_v) / (max_v - min_v)

    '''
    Replace class names with numbers
    '''
    labels = np.unique(df["class"])
    df["class"].replace({labels[0]: 0, labels[1]: 1, labels[2]: 2}, inplace=True)

    shuffle(df)

    segment_len = int(len(df) / folds)

    for k in range(3, k_max + 1, 2):
        print("k: ", k)

        for fold in range(0, folds - 1):
            validation_start = fold * segment_len

            hits = 0

            for validation in range(validation_start, validation_start + segment_len):
                v = df.iloc[validation]

                distances = list()
                for training in chain(range(0, validation_start),
                                      range(validation_start + segment_len, segment_len * (folds - 1))):
                    t = df.iloc[training]
                    distances.append((math.sqrt(
                        math.pow(t[0] - v[0], 2) + math.pow(t[1] - v[1], 2) +
                        math.pow(t[2] - v[2], 2) + math.pow(t[3] - v[3], 2)), training))

                distances.sort()

                results = [0] * len(labels)
                for c in range(0, k):
                    result = int(df.iloc[distances[c][1]]["class"])
                    results[result] += 1

                winner = 0
                for r in range(0, len(labels)):
                    if results[r] > winner:
                        winner = r

                if winner == v["class"]:
                    hits += 1

            print("fold ", fold, " accuracy: ", hits / segment_len)

        print()


def kNN_regressor():
    X, y = load_boston(return_X_y=True)

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
                        distance += math.pow(v[i]-t[i], 2)
                    distance = math.sqrt(distance)

                    distances.append((distance, training))

                distances.sort()

                nearestN = distances[:k]

                total_sum = 0.0
                for d in nearestN:
                    total_sum += d[0]

                Y = 0.0
                for n in nearestN:
                    Y += y[n[1]] * (n[0] / total_sum)

                mea += Y - y[validation]

                y_pred.append(Y)
                y_true.append(y[n[1]])

            error = mean_absolute_percentage_error(y_true, y_pred)
            print("fold ", fold, "error: ", error)




if __name__ == '__main__':
    kNN_classifier()
    kNN_regressor()
