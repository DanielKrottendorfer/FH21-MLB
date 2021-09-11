import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# ----------------------------------------------------------------------------------------------------------------------
def fn_plotDecisionSurface(xx1,xx2, Z, x1, x2,
                           points_color = True, y=None, cmap_=['b', 'orange', 'g'],
                           title="Iris Data", x_label = "", y_label = "",
                           show=True):

    '''
    put docstring here
    '''

    # createplot
    fig, ax = plt.subplots(1, 1)

    # set plotting area
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    ax.set_xlim((x1_min, x1_max))
    ax.set_ylim(x2_min, x2_max)

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    lcmap = ListedColormap(cmap_)
    ax.contourf(xx1, xx2, Z, cmap=lcmap, alpha=0.4)

    # Plot also the training points
    if points_color:
        ax.scatter(x1, x2, c=np.array(cmap_)[y.ravel()])
    else:
        ax.scatter(x1, x2, c="k")

    # annotate plot
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
    ax.set_title(title)
    if show:
        fig.show()


# ----------------------------------------------------------------------------------------------------------------------
def fn_kfoldStast(accs, prefix = "", precision=3, validation="", train_test="", verbose=True):

    '''
    put docstring here
    '''

    avg_ = np.mean(accs)
    std_ = np.std(accs)
    min_ = np.min(accs)
    max_ = np.max(accs)

    if verbose:
        print(prefix, "avg, std, min, max of accuracies: ",
              np.round(avg_,precision), np.round(std_, precision),
              np.round(min_,precision), np.round(max_,precision))

    return({"validation":validation, "train_test":train_test, "avg":avg_, "std":std_, "min":min_, "max":max_})


# ----------------------------------------------------------------------------------------------------------------------
def fn_validationProcedure(X,y, model, splitter, validation=""):

    '''
    put docstring here
    '''

    training_accuracies = []
    test_accuracies = []
    for train_idx, test_idx in splitter.split(X,y):

        # select train/test data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # train, fit, predict and evaluate
        model.fit(X_train, y_train)

        y_hat_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, y_hat_train)
        training_accuracies.append(acc_train)

        y_hat_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_hat_test)
        test_accuracies.append(acc_test)

    # take average, sd, max/min
    train_stats = fn_kfoldStast(training_accuracies, "\ntraining set", validation=validation, train_test="train")
    test_stats = fn_kfoldStast(test_accuracies, "\ntest set", validation=validation, train_test="test")

    return(train_stats, test_stats)


# ----------------------------------------------------------------------------------------------------------------------
def fn_getBestParams(results, columns=["validation_accuracy", "k", "metric", "weight"]):

    results_df = pd.DataFrame(results, columns=columns)
    results_df.sort_values("validation_accuracy", inplace=True, ascending=False)
    best_params = results_df.iloc[0, :]

    return best_params


# ----------------------------------------------------------------------------------------------------------------------
def fn_retrain(X_train, X_val, y_train, y_val, best_params, X_test, y_test):

    X_train = np.vstack([X_train, X_val])
    y_train = np.hstack([y_train, y_val])
    knn_model = KNeighborsClassifier(n_neighbors=best_params.values[1], metric=best_params.values[2], weights=best_params.values[3])
    knn_model.fit(X_train, y_train)
    y_hat_test = knn_model.predict(X_test)
    acc_test = accuracy_score(y_test, y_hat_test)

    return(acc_test)
