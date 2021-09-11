########################################################################################################################
# DEMO 3 - kNN, distance metrics, weighting & parameter tuning
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# REMARKS
# ----------------------------------------------------------------------------------------------------------------------


'''
Stefan Lackner, 2021.08
Please note: This code is only meant for demonstration purposes, much could be captured in functions to increase re-usability
'''


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from MAI_AusgleichskursML_kNN_UTILS_202108_sl import *


# ----------------------------------------------------------------------------------------------------------------------
# GET THE DATA
# ----------------------------------------------------------------------------------------------------------------------


# get data
data = load_iris()
X = data.data
y = data.target
X,y = shuffle(X,y, random_state=42)
target_labels = data.target_names
feature_names = data.feature_names


# ----------------------------------------------------------------------------------------------------------------------
# DISTANCE METRICS
# ----------------------------------------------------------------------------------------------------------------------


'''
Here we will use some distance metrics as provided with scikit learn
check the following links:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

please note: even if we wont find the biggest differences here it does not mean that distance metrics are not relevant!
On the contrary, for many use cases finding the right metric is essential
There is even a whole subfield of ML which is devoted to s.c. distance metric learning

For illustration we will only use the first two dimensions(attributes) of the dataset
'''


# prepare values for plotting + meshgird for decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05), np.arange(x2_min, x2_max, 0.05))
XX = np.c_[xx1.ravel(), xx2.ravel()] # combines all xx,yy values to cover the whole data space (cartesian product)

# loop over metrics
y_hat_dict = {}
for metric in ["manhattan", "euclidean", "minkowski"]:
    knn_model = KNeighborsClassifier(n_neighbors=3, metric=metric)
    if metric == "minkowski":
        knn_model = KNeighborsClassifier(n_neighbors=3, metric=metric, p=3)
    knn_model.fit(X[:,:2], y)
    y_hat_dict[metric] = knn_model.predict(X[:,:2])

    Z = knn_model.predict(XX)
    fn_plotDecisionSurface(xx1, xx2, Z, X[:, 0], X[:, 1], True, y_hat_dict[metric],
                           x_label=feature_names[0], y_label=feature_names[1],
                           title="Iris Data, k=3 - " + metric)

# check differences manually
diff1 = y_hat_dict["manhattan"] != y_hat_dict["euclidean"] # 2 difference
diff2 = y_hat_dict["manhattan"] != y_hat_dict["minkowski"] # 2 difference
diff3 = y_hat_dict["euclidean"] != y_hat_dict["minkowski"] # no difference

# find differences via plot
fn_plotDecisionSurface(xx1, xx2, Z, X[diff1,0], X[diff1, 1], False,
                       x_label=feature_names[0], y_label=feature_names[1],
                       title="Iris Data, k=3 - " + metric, show=False)


# ----------------------------------------------------------------------------------------------------------------------
# WEIGHTING
# ----------------------------------------------------------------------------------------------------------------------


'''
Uncertain classification
scikit learn will use a random assignment when there is an equal number of data points for each class
the resulting decision surface will be nonsense
'''


# create data
ds1 = np.array([[0,0,0], [0,0.5,1], [2,2,2]])
ds1 = pd.DataFrame(ds1, columns=["x1", "x2", "y"])
X = ds1.iloc[:,:2]
y = ds1.iloc[:,2].astype(int)

# plot data
fig, ax = plt.subplots(1, 1)
ax.grid(c="lightgrey", alpha=0.5)
colormap = np.array(["b", "orange", "g"])
ax.scatter(ds1["x1"], ds1["x2"], c=colormap[y], zorder=2, s=80)
ax.scatter(0.5, 0.5, c="red", marker="s", s=100)
fig.show()

# predict new datapoint at 0.5, 0.5
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_model.fit(X,y)
y_hat_test = knn_model.predict(np.array([[.5,.5]]).reshape((1,2)))
y_hat_test # class 0 is assigned

# check training set predictions
y_hat_train = knn_model.predict(X)
y_hat_train # class 0 is always assigned, even on the training set!

# plot decision surface
x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05), np.arange(x2_min, x2_max, 0.05))
XX = np.c_[xx1.ravel(), xx2.ravel()]
Z = knn_model.predict(XX)
fn_plotDecisionSurface(xx1, xx2, Z, X.iloc[:,0], X.iloc[:,1], y=y,
                       title="Weighted kNN Example - no IDW used", x_label="x1", y_label="x2")


'''
for the given example, weighting by distance can help
'''


# predict new datapoint at 0.5, 0.5
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
knn_model.fit(X,y)
y_hat_test = knn_model.predict(np.array([[0.5,0.5]]).reshape((1,2)))
y_hat_test # class 2 is assigned

# check training set predictions
y_hat_train = knn_model.predict(X)
y_hat_train # class 0 is always assigned, even on the training set!

# with weighting the decision surface is much more meaningful
Z = knn_model.predict(XX)
fn_plotDecisionSurface(xx1, xx2, Z, X.iloc[:,0], X.iloc[:,1], y=y,
                       title="Weighted kNN Example - with IDW", x_label="x1", y_label="x2")


'''
to find randomly predicted points, on can use predict_proba()
this will return the probability of the prediction (simply the fraction of the classes within the neighbors for each point)
'''


# predict_proba() training data, unweighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_model.fit(X,y)
y_hat_train_proba = knn_model.predict_proba(X)
y_hat_train_proba # returns 1/3 for all

# predict_proba() training data, weighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
knn_model.fit(X,y)
y_hat_train_proba = knn_model.predict_proba(X)
y_hat_train_proba # returns 1 since the distance to one class is 0!

# predict_proba() test data, unweighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_model.fit(X,y)
y_hat_train_proba = knn_model.predict_proba(np.array([[0.5,0.5]]).reshape((1,2)))
y_hat_train_proba # same as for training data, all probabilities are 1/3

# predict_proba() test data, unweighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
knn_model.fit(X,y)
y_hat_train_proba = knn_model.predict_proba(np.array([[0.5,0.5]]).reshape((1,2)))
y_hat_train_proba # now, probabilities are weighted according to distance


'''
Now we will introduce conflicting data points to see what happens
Always test special/problematic cases with frameworks before you use them!
'''


# create data
ds1 = np.array([[0,0,0], [0,0,1], [0,0,2]])
ds1 = pd.DataFrame(ds1, columns=["x1", "x2", "y"])
X = ds1.iloc[:,:2]
y = ds1.iloc[:,2].astype(int)

# conflicting data points, unweighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_model.fit(X,y)
y_hat_train = knn_model.predict_proba(X)
y_hat_train # uniform distribution

# conflicting data points, weighted
knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
knn_model.fit(X,y)
y_hat_train = knn_model.predict_proba(X)
y_hat_train # uniform distribution, compare to definition of IDW in wikipedia

# conflicting data points, k=1
knn_model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
knn_model.fit(X,y)
y_hat_train = knn_model.predict_proba(X)
y_hat_train # random assignment!


# ----------------------------------------------------------------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------------------------------------------------------------


'''
For data pre-processing check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
We will both the standard and the min max scalers available in scikit learn
'''

# create data
np.random.seed(99)
x1 = np.random.random(100) * 10
x2 = np.random.random(100) + 5
X = np.hstack([x1.reshape((100,1)), x2.reshape((100,1))])


# plot data
fig, ax = plt.subplots(1,1)
ax.scatter(x1, x2)
ax.set_xlim(0, 10)
ax.set_ylim(0,10)
ax.set_title("Different scales of attributes")
fig.show()

# calculate mean distance
d_x1 = pairwise_distances(x1.reshape(-1,1))
d_x1_mean = np.mean(d_x1)
d_x1_mean

d_x2 = pairwise_distances(x2.reshape(-1,1))
d_x2_mean = np.mean(d_x2)
d_x2_mean

# use scaling, standardization
scaler = StandardScaler()
scaler.fit(X)
X_scaled_std = scaler.transform(X)

fig, ax = plt.subplots(1,1)
ax.scatter(X_scaled_std[:,0], X_scaled_std[:,1])
ax.set_title("Attributes, Standard Scaler")
fig.show()

# use scaling, max, min
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled_minmax = scaler.transform(X)

fig, ax = plt.subplots(1,1)
ax.scatter(X_scaled_minmax[:,0], X_scaled_minmax[:,1])
ax.set_title("Attributes, Min-Max Scaler")
fig.show()

# check difference
diff_abs = np.abs(X_scaled_std - X_scaled_minmax)
np.mean(diff_abs)
np.max(diff_abs)
np.min(diff_abs)


# ----------------------------------------------------------------------------------------------------------------------
# SCALED vs. NON SCALED IRIS DATA
# ----------------------------------------------------------------------------------------------------------------------


'''
Since attributes in the iris dataset have roughly the same scale, there is no effect of scaling is not strong
For other datasets the effect might be considerable - scaling is mandatory for kNN & numeric values!
'''


# get data
data = load_iris()
X = data.data
y = data.target
target_labels = data.target_names
feature_names = data.feature_names

print("\n")
for i in np.arange(20):
    X,y = shuffle(X,y) # shuffle & random_state don't work
    target_labels = data.target_names
    feature_names = data.feature_names

    # perform classification on the dataset without scaling
    knn_model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="uniform")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    knn_model.fit(X_train,y_train)
    y_hat_test = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_hat_test)
    acc

    # perform classification on the dataset with scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)
    X_scale_train, X_scale_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.33, random_state=42)
    knn_model.fit(X_scale_train,y_train)
    y_hat_scale_test = knn_model.predict(X_scale_test)
    acc_scale = accuracy_score(y_test, y_hat_scale_test)
    acc_scale

    if acc != acc_scale:
        print("random state: ", i, ", accuracy unscaled: ", acc, ", accuracy scaled: ", acc_scale)


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETER TUNING - OPTION 0, Holdout analog
# ----------------------------------------------------------------------------------------------------------------------


'''
HP-tuning option 0 is only here for educational purposes
You should consider using other approaches, e.g. option 1,2 or decisions based on domain knowledge
HP-tuning option 1 is better than option 0, but the final accuracy estimation is unstable
Be careful when reporting accuracies based on this approach
HP-tuning option 2 is the most stable, it can be computationally burdensome for larger datasets and more complex algorithms
check https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
'''


# get data
data = load_iris()
X = data.data
y = data.target
#X, y = shuffle(X, y, random_state=42)
target_labels = data.target_names
feature_names = data.feature_names


# assemble HP combinations for grid search
ks = [1,3,5,7,9]
metrics = ["euclidean", "manhattan", "minkowski"]
weighting = ["uniform", "distance"]
unique_combinations = []
for k in ks:
    for m in metrics:
        for w in weighting:
            unique_combinations.append((k, m, w))
unique_combinations


# ----------------------------------------------------------------------------------------------------------------------
# Option0, holdout analog
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=99, stratify=y_train)
results = []
for comb in unique_combinations:
    knn_model = KNeighborsClassifier(n_neighbors=comb[0], metric=comb[1], weights=comb[2])
    if comb[1] == "minkowski":
        knn_model = KNeighborsClassifier(n_neighbors=comb[0], metric=comb[1], weights=comb[2], p = 3)

    # validation
    knn_model.fit(X_train, y_train)
    y_hat_val = knn_model.predict(X_val)
    acc_val = accuracy_score(y_val, y_hat_val)
    results.append([acc_val, comb[0], comb[1], comb[2]])

# best params and retrain
best_params = fn_getBestParams(results)
acc = fn_retrain(X_train, X_val, y_train, y_val, best_params,X_test, y_test)
acc


# ----------------------------------------------------------------------------------------------------------------------
# Option 1, Holdout/Cross Validation Analog
X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(X,y, test_size=0.2, random_state=99, stratify=y)
results_outer = []
for comb in unique_combinations:
    knn_model = KNeighborsClassifier(n_neighbors=comb[0], metric=comb[1], weights=comb[2])
    if comb[1] == "minkowski":
        knn_model = KNeighborsClassifier(n_neighbors=comb[0], metric=comb[1], weights=comb[2], p = 3)

    skf = StratifiedKFold(n_splits = 5, shuffle=True,random_state=99)
    results_inner = []
    for train_idx_inner, val_idx_inner in skf.split(X_train_outer, y_train_outer):
        X_train_inner = X_train_outer[train_idx_inner]
        X_val_inner = X_train_outer[val_idx_inner]
        y_train_inner = y_train_outer[train_idx_inner]
        y_val_inner = y_train_outer[val_idx_inner]

        knn_model.fit(X_train_inner, y_train_inner)
        y_hat_val_inner = knn_model.predict(X_val_inner)
        acc_val_inner = accuracy_score(y_val_inner, y_hat_val_inner)

        results_inner.append(acc_val_inner)

    mean_acc_val_inner = np.mean(results_inner)
    results_outer.append([mean_acc_val_inner, comb[0], comb[1], comb[2]])

# best params and retrain
best_params = fn_getBestParams(results_outer)
knn_model = KNeighborsClassifier(n_neighbors=best_params.values[1], metric=best_params.values[2], weights=best_params.values[3])
knn_model.fit(X_train_outer, y_train_outer)
y_hat_test_outer = knn_model.predict(X_test_outer)
acc = accuracy_score(y_test_outer, y_hat_test_outer)
acc


# ----------------------------------------------------------------------------------------------------------------------
# Option 2, nested cross validation
results_outer = []
skf_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=99)
skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=99)
results_hps = []
for comb in unique_combinations:

    print("\n\n" + "-"*20, "\nprocessing parameters: ", str(comb))
    knn_model = KNeighborsClassifier(n_neighbors=comb[0], metric=comb[1], weights=comb[2])

    results_test = []
    counter_outer = 0
    for train_outer_idx, test_outer_idx in skf_outer.split(X,y):
        counter_outer += 1

        print("\nprocessing outer fold: ", counter_outer)
        X_train_outer = X[train_outer_idx]
        y_train_outer = y[train_outer_idx]
        X_test_outer = X[test_outer_idx]
        y_test_outer = y[test_outer_idx]

        results_val = []
        counter_inner = 0
        for train_inner_idx, val_inner_idx in skf_inner.split(X_train_outer, y_train_outer):
            counter_inner += 1

            print("processing inner fold: ", counter_inner)
            X_train_inner = X[train_inner_idx]
            y_train_inner = y[train_inner_idx]
            X_val_inner = X[val_inner_idx]
            y_val_inner = y[val_inner_idx]

            knn_model.fit(X_train_inner, y_train_inner)
            y_hat_val_inner = knn_model.predict(X_val_inner)
            acc_val_inner = accuracy_score(y_val_inner, y_hat_val_inner)
            results_val.append(acc_val_inner)

        mean_acc_val = np.mean(results_val)
        knn_model.fit(X_train_outer, y_train_outer)
        y_hat_test_outer = knn_model.predict(X_test_outer)
        acc_test_outer = accuracy_score(y_test_outer, y_hat_test_outer)
        results_test.append(acc_test_outer)

    mean_acc_test = np.mean(results_test)
    results_hps.append([mean_acc_val, mean_acc_test, comb[0], comb[1], comb[2]])

# best params and refit
best_params = fn_getBestParams(results_hps, columns=["validation_accuracy", "test_accuracy", "k", "metric", "weight"])
acc = best_params["test_accuracy"]
acc


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETER TUNING using scikit learn
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Option 2, nested cross validation
# Set up possible values of parameters t
p_grid = {"n_neighbors": [1, 3, 5, 7, 9],
          "metric": ["euclidean", "manhattan"],
          "weights":["uniform", "distance"]}

# create model and k-fold iterators. Any CV-iterator could be used here!
knn_model = KNeighborsClassifier()
inner_cv = KFold(n_splits=4, shuffle=True, random_state=99)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=99)

# define GridSearch class & fit (fit is only to get info on best_params_ etc
clf = GridSearchCV(estimator=knn_model, param_grid=p_grid, cv=inner_cv, refit=True)
clf.fit(X,y)

# use GridSearch class inside cross_val_score
# clf defined via GridSearchCV is called inside cross_val score, meaning that split from xval will be split again by passed GridSearchCV
# results can be checked using cv_results_
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print(clf.best_params_)
print(nested_score.mean())
print(clf.cv_results_)
print(clf.cv_results_["mean_test_score"])
print(clf.cv_results_.keys())


# ----------------------------------------------------------------------------------------------------------------------
# Option 2, nested cross validation using RandomSearch
# create model and k-fold iterators. Any CV-iterator could be used here!
knn_model = KNeighborsClassifier()
inner_cv = KFold(n_splits=4, shuffle=True, random_state=99)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=99)

# define GridSearch class & fit (fit is only to get info on best_params_ etc
clf = RandomizedSearchCV(estimator=knn_model, param_distributions=p_grid, cv=inner_cv, n_iter=10)
clf.fit(X,y)

# use GridSearch class inside cross_val_score
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print(clf.best_params_)
print(nested_score.mean())














