########################################################################################################################
# DEMO 1
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# REMARKS
# ----------------------------------------------------------------------------------------------------------------------


'''
Stefan Lackner, 2021.08
Please note: This code is only meant for demonstration purposes, much could be captured in functions to increase re-usability
'''


'''
in this section we will walk trough a nearly complete supervised machine learning workflow
however, we wont go into details on e.g. how data is best split into train/validation/test sets.
details on all steps of the workflow will follow in later demos/sessions
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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from MAI_AusgleichskursML_kNN_UTILS_202108_sl import *


# ----------------------------------------------------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------------------------------------------------


'''
to follow along it is recommended that you set a data and a code folder for each session
collect all data and scripts in this directories
define a path variable pointing to your data folder
'''

path = os.getcwd()


# ----------------------------------------------------------------------------------------------------------------------
# GET LABELED DATA
# ----------------------------------------------------------------------------------------------------------------------


'''
In this example we will use a prepared iris dataset
for this example you can assume that everything is ok with the data
we wont go into details regarding data handling (next session)
we don't need to care about class balancing since the data set has 3 classes with ~50 examples of each class

important note:
- to simulate "going into production" the prepared dataset has rows with and without labels
- rows without labels represent "production data"
- usually, when you organize/receive a dataset for training a supervised learning algorithm, all data points must have associated labels

important note:
- the iris dataset has 4 features
- for illustration purposes we will only use the first two features for training an algorithm 
'''


# load and inspect the data
df = pd.read_csv(os.path.join(path, "iris_missinglabels.csv"), sep =";")
print("\n", type(df))
df.info()
print("\n", df["class"])

# look at the dataframe, set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5) # only relevant if there is a sufficient precision in the data
print(df)

# split dataset into parts with and without labels
na_idx = df["class"].isna()
print("\n", na_idx)
df_labels = df[~na_idx]
df_labels.info()
df_nolabels = df[na_idx]
df_nolabels.info()

# replace labels (names) with integer values (needed for scikit learn)
labels = np.unique(df_labels["class"])
df_labels["class"].replace({labels[0]: 0, labels[1]: 1, labels[2]: 2}, inplace=True)


# ----------------------------------------------------------------------------------------------------------------------
# DATA EXPLORATION & Cleaning
# ----------------------------------------------------------------------------------------------------------------------


'''
to see how classes overlap or separate in data space on can use pair plots
here we only use the seaborn pairplot for data investigation
of course, many more options for data exploration are available and should be used
'''


sns.pairplot(df_labels, hue="class")


'''
We see that there is some problem with the data
obviously there is at least one corrupt datapoint with suspiciously high values
we need to exclude this data point before analysis
'''


df_labels.head()
df_labels.drop(axis=1, index=0, inplace=True)
sns.pairplot(df_labels, hue="class")


'''
After cleaning we extract X,y from the data frame
the scikit learn API always needs X,y to be handed over separately
'''


X = df_labels.iloc[:,:2]
y = df_labels.iloc[:,4]
X_pred = df_nolabels.iloc[:,:2]


# ----------------------------------------------------------------------------------------------------------------------
# CHOOSE AN ALGORITHM AND POSSIBLE HYPER-PARAMETER SETTINGS
# ----------------------------------------------------------------------------------------------------------------------


'''
we will choose the kNN classifier in this demo
There is no specific data dependent reason, kNN is just easy and good for an ML introduction
As Hyper-Parameters we chose k, the number of neighbors to be either 3 or 11
'''


'''
links:
https://scikit-learn.org/stable/modules/classes.html
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
https://scikit-learn.org/stable/modules/neighbors.html#classification
'''


# ----------------------------------------------------------------------------------------------------------------------
# DATA PREPROCESSING
# ----------------------------------------------------------------------------------------------------------------------


'''
in this workflow we won't perform any feature engineering but keep in mind that this is an important step
Usually for kNN scaling of attributes is performed (more later)
'''


# ----------------------------------------------------------------------------------------------------------------------
# DATA SPLITTING
# ----------------------------------------------------------------------------------------------------------------------


'''
Here we will use a simple repeated holdout validation to create train/validation and test sets
More details on how performance assessment can be done reliably will follow
Please keep in mind that the approach present here is easy but not the best one!
'''


X_train1, X_test, y_train1, y_test = train_test_split(X, # the features
                                                      y, # the labels/target values
                                                      test_size=0.2, # size of the test set set aside, usually 0.2 or 0.33
                                                      shuffle=True, # if the dataset should be shuffled before splitting
                                                      random_state=99, # random seed that ensures we#ll always get the same results
                                                      stratify=y) # startification by labels (label distribution is kept in both partitions)


'''
Now we have split the whole dataset with labels into a training and a test set
we need a validation set also. To get one we'll split the training set again into a training and a validation set
'''


X_train2, X_val, y_train2, y_val = train_test_split(X_train1,
                                                    y_train1,
                                                    test_size=0.2, # no shuffle is nescessary as this has been done before
                                                    random_state=99,
                                                    stratify=y_train1)



# ----------------------------------------------------------------------------------------------------------------------
# TRAIN THE ALGORITHM AND TUNE (SELECT HYPER-PARAMETERS)
# ----------------------------------------------------------------------------------------------------------------------


'''
please note: kNN is a good start because its simple, but it does not really learn a model!
kNN is a s.c. lacy learning approach where you'll always need to keep the whole dataset!
the one thing being closest to "a model", is the s.c. voronoi tesselation of the data space
However, we don't have any parameters fitted to the data but only hyper parameters!
'''


'''
workflow:
1) Instantiate the class of the algorithm you want to use with the chosen hyper-parameters (and possibly other arguments)
2) fit the algorithm (function) to the data
3) use the fitted algorithm (function) to predict points with unknown y values (labels/targets)
'''


'''
please note: even if preformance assessment is coneptuall a different step, we need to measure performance in this step
to choose the best hyper-parameter setting we know which one perfoms better
here we use a simple accuracy measure (# correct predictions/ # all predictions)
'''


# knn with hyperparameter k = 5
knn_model = KNeighborsClassifier(n_neighbors=3, # hyperparameter k, usually odd, thehigher the smoother the decision surface
                                 metric="euclidean", # metric used for distance calculation (more later)
                                 weights="uniform") # if estimation should be weighted by distance (more later)
knn_model = knn_model.fit(X_train2, y_train2)
y_hat_val_k5 = knn_model.predict(X_val)
acc_val_k5 = accuracy_score(y_val, y_hat_val_k5)
print("\naccuracy on the validation set with k=5: ", acc_val_k5)

# knn with hyperparameter k = 11
knn_model = KNeighborsClassifier(n_neighbors=11, # hyperparameter k, usually odd, thehigher the smoother the decision surface
                                 metric="euclidean", # metric used for distance calculation (more later)
                                 weights="uniform") # if estimation should be weighted by distance (more later)
knn_model = knn_model.fit(X_train2, y_train2)
y_hat_val_k11 = knn_model.predict(X_val)
acc_val_k11 = accuracy_score(y_val, y_hat_val_k11)
print("\naccuracy on the validation set with k=11: ", acc_val_k11)


'''
in this case k=11 is seems to be the better choice, so we#ll use it to do perfoamce assessment
please note again: using 2 simple splits to create train, validation and test sets is not recommended
better options will follow later
'''


# ----------------------------------------------------------------------------------------------------------------------
# PERFORMANCE EVALUATION
# ----------------------------------------------------------------------------------------------------------------------


'''
Now we know that - given the validation procedure an hyper-parameter candidates we chose! - what the best settings are
Since the training and the validation set where used for selecting hyper-parameters, we cannot assess the performance on them
we need an independent set, which was not used before to ensure unbiased performance estimates!
using performance measures from the validation set would be data snooping!
'''


'''
first we retrain the algorithm with the best hyper-parameter setting on the test and the validation set
this has been set aside as X_train1 and y_train1 in section data splitting
'''


# knn with hyperparameter k = 11
knn_model = KNeighborsClassifier(n_neighbors=11, # hyperparameter k, usually odd, thehigher the smoother the decision surface
                                 metric="euclidean", # metric used for distance calculation (more later)
                                 weights="uniform") # if estimation should be weighted by distance (more later)
knn_model = knn_model.fit(X_train1, y_train1)
y_hat_test_k11 = knn_model.predict(X_test)
acc_test_k11 = accuracy_score(y_test, y_hat_test_k11)
acc_val_k11
print("\naccuracy on the test set with k=11: ", acc_val_k11)


# ----------------------------------------------------------------------------------------------------------------------
# PRESENTING RESULTS
# ----------------------------------------------------------------------------------------------------------------------


'''
in this section we will plot decision surfaces to see how our algorithm separates the data space to make predictions
Here we ill also look at out fake-production data (the data without labels)
usually you don't have production data until you activate your system
'''


# plot data which was used for fitting (only the first 2 attributes)
fig, ax = plt.subplots(1, 1)
x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
labels = y.values
colormap = np.array(["b", "orange", "g"])
ax.scatter(X.iloc[:,0], X.iloc[:,1], c=colormap[labels])
ax.set_ylabel(X.columns[0])
ax.set_xlabel(X.columns[1])
ax.set_title("Iris Data")
ax.set_xlim((x1_min, x1_max))
ax.set_ylim(x2_min, x2_max)
fig.show()


# create meshgrid for decision boundary/surface visualization
# see https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05), np.arange(x2_min, x2_max, 0.05))
XX = np.c_[xx1.ravel(), xx2.ravel()] # combines all xx,yy values to cover the whole data space (cartesian product)
Z = knn_model.predict(XX) # use the previously fitted algorithm with the best HP-settings


'''
to illustrate how our algorithm behaves, we will plot
(1) the decision surface with data points used for training (after tuning)
(2) the decision surface with data points used for testing
'''


fn_plotDecisionSurface(xx1, xx2, Z, X_train1.iloc[:,0], X_train1.iloc[:,1], True, y_train1,
                       x_label=df_labels.columns[0], y_label=df_labels.columns[1],
                       title="Iris Data - Decision Surface & Data Points used for Training")
fn_plotDecisionSurface(xx1, xx2, Z, X_test.iloc[:,0], X_test.iloc[:,1], True, y_test,
                       x_label=df_labels.columns[0], y_label=df_labels.columns[1],
                       title="Iris Data - Decision Surface & Data Points used for Testing")


'''
finally, we will illustrate the behaviour of our algorithm on the fake-production data
'''


fn_plotDecisionSurface(xx1, xx2, Z, df_nolabels.iloc[:,0], df_nolabels.iloc[:,1], False,
                       x_label=df_labels.columns[0], y_label=df_labels.columns[1],
                       title="Iris Data - Decision Surface & 'Fake-Production' Data")

plt.show()
