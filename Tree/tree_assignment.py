

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def tree_assignment():
    (X, y) = load_iris(return_X_y=True)

    iX = list()
    iy = list()

    for _x, _y in zip(X[:, 0], y):
        if _y == 1:
            continue

        iX.append(_x)
        iy.append(_y)

    iX, iy = shuffle(iX, iy, random_state=42)
    iX, X_test, iy, y_test = train_test_split(iX, iy, test_size=0.2)
    lowest_gi = 1
    best_separator = iX[0]
    size = len(iX)

    for i in range(0, size, 2):
        a = (iX[i] + iX[i+1])/2

        j1, n1 = (0, 0)
        j2, n2 = (0, 0)

        for _x, _y in zip(iX, iy):
            if _x < a:
                if _y == 0:
                    j1 += 1
                else:
                    n1 += 1
            else:
                if _y == 0:
                    j2 += 1
                else:
                    n2 += 1

        s, z = (j1+n1, j2+n2)
        g1 = 1 - (j1/s)**2 - (n1/s)**2
        g2 = 1 - (j2/z)**2 - (n2/z)**2

        g = (s/(s+z)*g1) + (z/(s+z)*g2)
        if lowest_gi > g:
            lowest_gi = g
            best_separator = a

    tp, fp, tn, fn = 0, 0, 0, 0
    for _x, _y in zip(X_test, y_test):
        if _x < best_separator:
            if _y == 0:
                tp += 1
            else:
                fp += 1
        else:
            if _y == 0:
                fn += 1
            else:
                tn += 1

    print("best separator: ", best_separator, " gini: ",lowest_gi)

    print(tp, " ", fn, " ", tn, " ", fn)

    acc = (tp+tn) / len(y_test)

    print(acc)


if __name__ == "__main__":
    tree_assignment()
