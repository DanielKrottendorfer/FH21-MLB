########################################################################################################################
# EX1
########################################################################################################################


'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutff value for idnetifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers!
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''

########################################################################################################################
# Solution
########################################################################################################################


# set pandas options to make sure you see all info when printing dfs
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_wine
from scipy.stats import zscore
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

path = os.getcwd()
df = pd.read_csv(os.path.join(path, "wine_exercise.csv"), sep=";", header=1, skipfooter=1, engine="python")

'''
1
'''
for i in range(0, len(df)):
    if df["country-age"].iloc[i] is not None:
        continue

    column = df.iloc[i]
    row_str = str(column[0])
    commas = row_str.count(',')
    row_splits = row_str.split(',')

    if len(column) - commas > 1:
        for y in range(1, len(column) - commas):
            df.iloc[i, y + commas] = column[y]

    for y in range(commas, -1, -1):
        df.iloc[i, y] = row_splits.pop()

for i, column in df.iterrows():
    for y in range(0, len(column)):
        column[y] = str(column[y]).replace(',', '.')
        column[y] = str(column[y]).replace(' ', '')
        if str(column[y]).startswith('nan') | str(column[y]).startswith('missing'):
            column[y] = np.nan

df["alcohol"] = pd.to_numeric(df["alcohol"])
df["malic_acid"] = pd.to_numeric(df["malic_acid"])
df["ash"] = pd.to_numeric(df["ash"])
df["alcalinity_of_ash"] = pd.to_numeric(df["alcalinity_of_ash"])
df["magnesium"] = pd.to_numeric(df["magnesium"])
df["total_phenols"] = pd.to_numeric(df["total_phenols"])
df["flavanoids"] = pd.to_numeric(df["flavanoids"])
df["nonflavanoid_phenols"] = pd.to_numeric(df["nonflavanoid_phenols"])
df["proanthocyanins"] = pd.to_numeric(df["proanthocyanins"])
df["color_intensity"] = pd.to_numeric(df["color_intensity"])
df["hue"] = pd.to_numeric(df["hue"])
df["od280/od315_of_diluted_wines"] = pd.to_numeric(df["od280/od315_of_diluted_wines"])
df["proline"] = pd.to_numeric(df["proline"])
df["target"] = pd.to_numeric(df["target"])
df["color"] = pd.to_numeric(df["color"],downcast="integer")

'''
2
'''
df = df.drop_duplicates()

'''
3
'''
for column in ["magnesium", "total_phenols"]:
    ps = df[column]
    ps = ps.dropna()
    ps = zscore(ps)
    for i in ps.keys():
        p = (2 * (1 - ps[i]))
        if p < 0.05:
            df[column].iloc[i] = np.nan

for i in range(0, len(df["malic_acid"])):
    if df["malic_acid"].iloc[i] == -999.0:
        df["malic_acid"].iloc[i] = np.nan

'''
4
'''
wine = load_wine()

imp = KNNImputer(n_neighbors=3)
imp.fit(wine.data)
imputed_data = imp.transform(df.iloc[:, :13].values)

for i in range(0, len(imputed_data)):
    for y in range(0, 13):
        df.iloc[i, y] = imputed_data[i, y]

'''
5
'''
ax = df.groupby(["target"]).size().plot(kind="bar")

'''
6
'''
print(df[["magnesium", "color"]].groupby(["color"], as_index=False).agg(["count","mean","std"]))


plt.show()
