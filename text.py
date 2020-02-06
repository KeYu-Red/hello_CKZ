import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basic_func as func
import time
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from nltk.stem import PorterStemmer


(listing_id, features, values, train_df) = func.load_unicef_data(fname="train.json", data=None)

high = train_df.loc[train_df["interest_level"]=="high", ["listing_id"]]
high_train = train_df.loc[train_df["listing_id"].isin(high["listing_id"])]
(listing_id, features, values_high, high_train) = func.load_unicef_data(data=high_train)

low = train_df.loc[train_df["interest_level"]=="low", ["listing_id"]]
low_train = train_df.loc[train_df["listing_id"].isin(low["listing_id"])]
(listing_id, features, values_low, low_train) = func.load_unicef_data(data=low_train)


p = features.index('description')
corpus_high = values_high[:, p]
corpus_low = values_low[:, p]
print(values_high.shape)
print(values_low.shape)

for i in range(len(corpus_high)):
    corpus_high[i] = ''.join(filter(func.not_digit_and_underline, corpus_high[i]))
for i in range(len(corpus_low)): 
    corpus_low[i] = ''.join(filter(func.not_digit_and_underline, corpus_low[i]))
# print(corpus[1])
# max_df and min_df can be determined by cross validation
ps = PorterStemmer()
vectorizer_high = CountVectorizer(max_df= 0.2, min_df=0.03, ngram_range=(1, 2), stop_words='english')
vectorizer_low = CountVectorizer(max_df= 0.2, min_df=0.03, ngram_range=(1, 2), stop_words='english')

X_high = vectorizer_high.fit_transform(corpus_high)
X_low = vectorizer_low.fit_transform(corpus_low)
feature_name_high = vectorizer_high.get_feature_names()
feature_name_low = vectorizer_low.get_feature_names()
print(feature_name_high)
print(len(feature_name_high))
print(feature_name_low)
print(len(feature_name_low))
for i in range(len(feature_name_high)):
    feature_name_high[i] = ps.stem(feature_name_high[i])
for i in range(len(feature_name_low)):
    feature_name_low[i] = ps.stem(feature_name_low[i])
# print(set(feature_name_high))
# print(len(set(feature_name_high)))
# print(set(feature_name_low))
# print(len(set(feature_name_low)))


# print(feature_name[166])