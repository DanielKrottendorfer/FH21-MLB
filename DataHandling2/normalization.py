import math
import os
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

folds = 4 + 1
k_max = 11


def kNN_classifier(normalize):
    X_y = load_iris()
    (X, y) = shuffle(X_y["data"], X_y["target"], random_state=0)

    X[:, 0] *= 100

    '''
    Normalize data
    '''
    if normalize:
        X = preprocessing.normalize(X, norm='max', axis=0)

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


if __name__ == '__main__':
    kNN_classifier(normalize=0)
    print("-------------------------------------")
    kNN_classifier(normalize=1)
