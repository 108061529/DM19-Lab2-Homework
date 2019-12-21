# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:26:39 2019

@author: menghan
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## load a pickle file
test_df = pd.read_pickle("test_public_tweets.pkl")

with open('D:/DMLAB/DMLab2/DM19-Lab2-Homework/models/' +  'lin_svm_balance_train_128000_with_tfidf.pkl', 'rb') as f: #####################
    clf = pickle.load(f)
y_pred = clf.predict(test_df['text'])
y_pred = list(y_pred)
df_submit = pd.DataFrame()
df_submit['id'] = test_df.id
df_submit['emotion'] = y_pred
df_submit.to_csv('submit/submit_model_lin_svm_balance_train_128000_with_tfidf.csv', index=False)    
