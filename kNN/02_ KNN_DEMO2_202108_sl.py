########################################################################################################################
# DEMO 2
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# REMARKS
# ----------------------------------------------------------------------------------------------------------------------


'''
Stefan Lackner, 2021.08
Please note: This code is only meant for demonstration purposes, much could be captured in functions to increase re-usability
'''


'''
This time we use the whole iris dataset and load it via the scikit learn datasets moduls
.fit() will be used on the training sets, .predict() will be used on the test sets
We will use accuracy as a simple performance measure for classification
There are many more performance measures, espacially for classification, where class balancing plays an important role
In the iris data set we have 150 data points with 50 data points per class
Check: https://scikit-learn.org/stable/modules/cross_validation.html 
'''


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------


'''
To follow this script you´ll need the following packages
- numpy
- pandas
- sklearn (scikit-learn)
- matplotlib, seaborn
They can be installed via the Anaconda Distribution or via PIP 
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, ShuffleSplit, \
    StratifiedShuffleSplit, LeaveOneOut, LeavePOut, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from MAI_AusgleichskursML_kNN_UTILS_202108_sl import *


# ----------------------------------------------------------------------------------------------------------------------
# LOAD THE IRIS DATASET
# ----------------------------------------------------------------------------------------------------------------------


'''
This time we will load the complete, unchanged iris data from scikit learn
'''


# look at dss.load_ + tab to see available datasets
data = load_iris()
print(data.keys())
print(data.DESCR)

# get as X and y
X = data.data
y = data.target

# shuffle
X, y = shuffle(X, y, random_state=99) # pease note: random state does not work with shuffle()

# combine X and y into one matrix
y_rs = y.reshape((y.shape[0], 1))
Xy = np.hstack([X,y_rs])
Xy.shape

# get target labels/feature names
target_labels = data.target_names
feature_names = data.feature_names

# create pandas df, this is just for having the data in a more convenient format for e.g. value replacement
df = pd.DataFrame(Xy, columns = feature_names + ["class"])
df["class"].replace({0:target_labels[0], 1:target_labels[1], 2:target_labels[2]}, inplace=True)


# ----------------------------------------------------------------------------------------------------------------------
# PERFORMANCE METRICS, CONFUSION MATRICES, ROC-CURVES
# ----------------------------------------------------------------------------------------------------------------------


'''
in this section we will select only two classes from the iris dataset
we will not split the data since we only want to demonstrate the use of confusion matrices
we wont do any parameter tuning either
please note that the performance we receive here is the s.c. in-sample or training error/accuracy
never forget, to do hyper-parameter tuning and performance evaluation properly, we need to split our data!
'''


# select classes 0,1
cl12_idx = (y == 1)|(y == 2)
X = X[cl12_idx]
X = X[:,:2]
y = y[cl12_idx]

# train the model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X,y)
y_hat = knn_model.predict(X)
acc = accuracy_score(y, y_hat)

# create a confusion matrix, normalized and not normalized
cm = confusion_matrix(y, y_hat) # true labels in rows, predicted in columns
cmn = confusion_matrix(y, y_hat, normalize="all")

# plot confusion matrices
plot_confusion_matrix(knn_model, X, y, cmap=plt.cm.Blues, display_labels=[target_labels[1], target_labels[2]])
plot_confusion_matrix(knn_model, X, y, cmap=plt.cm.Blues, display_labels = [target_labels[1], target_labels[2]], normalize="all")

# get accuracy scores via scikit learn
acc = accuracy_score(y, y_hat)
acc_bal = balanced_accuracy_score(y, y_hat) # since classes are balanced, this should be equal to acc
f1 = f1_score(y, y_hat)
recall_tp_sensitivity = recall_score(y, y_hat)
precision_tn_specifity = precision_score(y, y_hat)

# get TP, TN, FP, FN and specifity from cm
TPR_recall_sensitivity = cm[0,0]/cm.sum(axis=1)[0]
FNR = cm[0,1]/cm.sum(axis=1)[0]
FPR = cm[1,0]/cm.sum(axis=1)[1]
TNR_specifity = cm[1,1]/cm.sum(axis=1)[1]
PPV_precision = cm[0,0]/cm.sum(axis=0)[0]


'''
now we will change our data s.t. classes are unbalanced
unbalanced classes are important for both fitting an algorithm and for evaluating performance
in a multiclass problem, if one class is overrepresented and well predicted, accuracy will always be good
bad results on underreprestend classes are glossed over, but this might not be well aligned with the problem 
'''


# select cases where sepal length < 6.5
sl6p5_idx = X[:,0] < 6.5
X_ = X[sl6p5_idx]
y_ = y[sl6p5_idx]
np.unique(y_, return_counts=True) # c1=41, c2=24

# train the model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_,y_)
y_hat = knn_model.predict(X_)
acc = accuracy_score(y_, y_hat)

# create a confusion matrix, normalized and not normalized
cm = confusion_matrix(y_, y_hat) # true labels in rows, predicted in columns
cmn = confusion_matrix(y_, y_hat, normalize="all")

# get performance measures
recall_tp_sensitivity = recall_score(y_, y_hat) # class 1 is predicted correctly with 0.78 probability
TNR_specifity = cm[1,1]/cm.sum(axis=1)[1] # class 2 is predicted correctly with 0.708 probability
acc = accuracy_score(y_, y_hat) # unweighted overall accuracy is 0.753
acc_bal = balanced_accuracy_score(y_, y_hat) # balanced accuracy is a little lowe since class 2 is under represented
acc_bal2 = (TNR_specifity + recall_tp_sensitivity)/2 # by hand. since TNR and TPR are already normalized the calcuation is easy


'''
we will switch to the balanced data again to illustrate the use of ROC-Curves on binary classification problems
ROC Curves plot TPR (Sensitivity) against FPR to illustrate the trade-off between the two
the larger the area under the roc curve, the better is the classifier at handling the problem, both from aTPR and a TNR viewpoint
please note: to create the ROC curve we need predicted probabilities
This can be done using predict_probability()
'''


# train the model
knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X,y)
y_hat_proba = knn_model.predict_proba(X)
y_hat_proba = y_hat_proba[:,1]

# calculate ROC values
fpr, tpr, thresholds = roc_curve(y, y_hat_proba, pos_label=2) # pos label is class 2, this mus be aligned with the selection from predict_proba()
roc_auc = roc_auc_score(y, y_hat_proba)

# plot
fig, ax = plt.subplots(1,1)
ax.grid(alpha=0.5)
ax.plot(fpr, tpr, c="darkorange", label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax.legend(loc="lower right")
ax.set_ylabel("TPR")
ax.set_xlabel("FPR")
ax.set_title("ROC-AUC, kNN with k=11 on Iris Data\nClasses 1,2 - Sepal Length/Width used for Prediction")
fig.show()


# ----------------------------------------------------------------------------------------------------------------------
# RE-SUBSTITUTION EVALUATION
# ----------------------------------------------------------------------------------------------------------------------


# load the data & shuffle
data = load_iris()
X = data.data
y = data.target
X, y = shuffle(X, y, random_state=99) # pease note: random state does not work with shuffle()

# instantiate knn class, fit & predict
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_model = knn_model.fit(X, y)
y_hat = knn_model.predict(X)

# get accuracy
acc = accuracy_score(y, y_hat)
print("\n", "accuracy score resubstitution evaluation: ", acc)

# get missed data points
missed_idx = y != y_hat
df_missed = df.loc[missed_idx, :]
y_hat[missed_idx] # confusion between classes 2 & 1
y[missed_idx]

# plot wrong classifications (remember, this DS has 4 dimensions!)
# plot data for fitting
fig, ax = plt.subplots(1, 1)
labels = y
colormap = np.array(["b", "orange", "g"])
ax.scatter(df_missed.iloc[:,0], df_missed.iloc[:,1], marker="s", s=100, c="r") # shown is ground truth
ax.scatter(df.iloc[:,0], df.iloc[:,1], c=colormap[labels])
ax.set_ylabel(df.columns[0])
ax.set_xlabel(df.columns[1])
ax.set_title("Iris Data, Wrong Classifications")
fig.show()


# ----------------------------------------------------------------------------------------------------------------------
# 1-Fold Cross-Validation (Holdout Validation)
# ----------------------------------------------------------------------------------------------------------------------


# instantiate knn class,
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# create train test split using train_test_split (returns data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)

# fit on the training data
knn_model.fit(X_train, y_train)

# predict on the test data & evaluate performance
y_hat_test = knn_model.predict(X_test)
acc_test = accuracy_score(y_test, y_hat_test)
print("\n", "accuracy score holdout evaluation, test-set: ", acc_test)

# check prediction on the training set
y_hat_train = knn_model.predict(X_train)
acc_train = accuracy_score(y_train, y_hat_train)
print("\n", "accuracy score holdout evaluation, training-set: ", acc_train)


# ----------------------------------------------------------------------------------------------------------------------
# k-Fold Cross-Validation (Holdout Validation), VERSION 1 using KFold
# ----------------------------------------------------------------------------------------------------------------------


# instantiate knn class,
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# instantiate KFold class
kf = KFold(n_splits=5, shuffle=True, random_state=99)

# fire validation procedure
train_stats_kfold, test_stats_kfold = fn_validationProcedure(X,y, knn_model, kf, "KFold")


# ----------------------------------------------------------------------------------------------------------------------
# k-Fold Cross-Validation (Holdout Validation), VERSION 2 cross_val_score
# ----------------------------------------------------------------------------------------------------------------------


knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
accs = cross_val_score(knn_model, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (accs.mean(), accs.std()))


# ----------------------------------------------------------------------------------------------------------------------
# k-Fold Cross-Validation (Holdout Validation), VERSION 3 using stratified k-fold
# ----------------------------------------------------------------------------------------------------------------------


'''
in the iris data set classes are evenly distributed
in many data sets classes are imbalanced however
in this case one mus ensure that class distributions are the same in tran/test sets
stratified kfold does exactly this
please note: if you don't stratify, you pipelines might crash id a class that's nt in train is in test!
Q: how do you deal with he case where - for at least one class - there are less class members than folds?
'''


stkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
train_stats, test_stats = fn_validationProcedure(X,y, knn_model, stkf, "StratifiedKFold")


# ----------------------------------------------------------------------------------------------------------------------
# Monte Carlo Cross Validation - aka Shuffle Split, V1
# ----------------------------------------------------------------------------------------------------------------------


ss = ShuffleSplit(n_splits=5, random_state=99)
train_stats_mccv, test_stats_mccv = fn_validationProcedure(X,y, knn_model, ss, "ShuffleSplit")



# ----------------------------------------------------------------------------------------------------------------------
# Monte Carlo Cross Validation - aka Shuffle Split, V2 - stratified
# ----------------------------------------------------------------------------------------------------------------------


sss = StratifiedShuffleSplit(n_splits=5, random_state=99)
train_stats, test_stats = fn_validationProcedure(X,y, knn_model, sss, "StratifiedShuffleSplit")


# ----------------------------------------------------------------------------------------------------------------------
# leave one out cross validation
# ----------------------------------------------------------------------------------------------------------------------


# instantiate stratified LOOCV class
loo = loo = LeaveOneOut()
training_accuracies = []
test_accuracies = []
for train_idx, test_idx in loo.split(X):

    # select train/test data
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # fit/predict/accuracies
    knn_model.fit(X_train, y_train)
    y_hat_train = knn_model.predict(X_train)
    acc_train = accuracy_score(y_train, y_hat_train)
    training_accuracies.append(acc_train)

    y_hat_test = knn_model.predict(X_test)
    acc_test = accuracy_score(y_test, y_hat_test)
    test_accuracies.append(acc_test)

    # take average, sd, max/min
train_stats_loocv = fn_kfoldStast(training_accuracies, "\ntraining set", validation="LeaveOneOut", train_test="train")
test_stats_loocv = fn_kfoldStast(test_accuracies, "\ntest set", validation="LeaveOneOut", train_test="test")


# ----------------------------------------------------------------------------------------------------------------------
# leave p out cross validation
# ----------------------------------------------------------------------------------------------------------------------


'''
use class LeavePOut from sklearn.model_selection
'''


# ----------------------------------------------------------------------------------------------------------------------
# plot accuracies
# ----------------------------------------------------------------------------------------------------------------------


df_acc_stats = pd.DataFrame([train_stats_kfold, test_stats_kfold, train_stats_mccv, test_stats_mccv, train_stats_loocv, test_stats_loocv])
fig, axs = plt.subplots(2, 1)
fig.tight_layout()
train_idx = df_acc_stats["train_test"] == "train"
test_idx = df_acc_stats["train_test"] == "test"
x_vals = np.array([0,1,2])

axs[0].plot(x_vals, df_acc_stats.loc[train_idx, "avg"].values, c = "b")
axs[0].plot(x_vals,df_acc_stats.loc[train_idx, "avg"].values + df_acc_stats.loc[train_idx, "std"].values,  c= "b", linestyle="dashed")
axs[0].plot(x_vals,df_acc_stats.loc[train_idx, "avg"].values - df_acc_stats.loc[train_idx, "std"].values,  c = "b", linestyle="dashed")
axs[0].set_xticks([0,1,2])
axs[0].set_xticklabels(["KFold", "ShuffleSplit", "LeaveOneOut"])
axs[0].set_ylim((0.8, 1.0))

axs[1].plot(x_vals, df_acc_stats.loc[test_idx, "avg"], c = "r")
axs[1].plot(x_vals, df_acc_stats.loc[test_idx, "avg"] + df_acc_stats.loc[test_idx, "std"], c= "r", linestyle="dashed")
axs[1].plot(x_vals, df_acc_stats.loc[test_idx, "avg"] - df_acc_stats.loc[test_idx, "std"], c = "r", linestyle="dashed")
axs[1].set_xticks([0,1,2])
axs[1].set_xticklabels(["KFold", "ShuffleSplit", "LeaveOneOut"])
axs[1].set_ylim((0.8, 1.0))

axs[0].set_title("training accuracy (mean, +/- 1 std)")
axs[1].set_title("testing accuracy (mean, +/- 1 std)")

fig.show()


########################################################################################################################
# DEMO 3 - kNN, distance metrics, weighting & parameter tuning
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# DISTANCE METRICS
# ----------------------------------------------------------------------------------------------------------------------


# get data
data = load_iris()
X = data.data
y = data.target
X,y = shuffle(X,y)
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
y_hat_tets = knn_model.predict(np.array([[.5,.5]]).reshape((1,2)))
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
# DATA SCALING, A TUTORIAL EXAMPLE
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








