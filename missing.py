import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import basic_func as func

# ---------missing values--------------
# building_id '0' 8286 ignore
# created  "" 0
# description "" 1446 ignore it/None
# display_address "" 135
# features [] 3218
# latitude number 0
# listing_id number 0
# longitude number 0
# manager_id str 0
# photos [] 3615 ignore
# price number 0
# street_address "" 10 drop
# interest_level 0

# -----------outlier--------------


(listing_id, features, values) = func.load_unicef_data()
train_df = pd.read_json("train.json")
# record the number of missing values
# initialize the counts
counts = dict()
for feature in features:
    counts[feature] = 0
# print(type(values[1, p]) is int)
print(train_df.isnull().sum())
for p in range(0, len(features)):
    for i in range(values[:, p].shape[0]):
        # missing values of building_id' is "0"
        if features[p] == "building_id" and values[i, p] == "0":
            counts[features[p]] = counts[features[p]] + 1
        # missing value of string types
        if type(values[1, p]) is str:
            if values[i, p] == "":
                counts[features[p]] = counts[features[p]]+1
        # missing value of list value
        if type(values[1, p]) is list:
            if not values[i, p]:
                counts[features[p]] = counts[features[p]]+1
print(counts)


