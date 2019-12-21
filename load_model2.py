# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:41:53 2019

@author: menghan
"""

from keras.models import load_model

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
## load a pickle file
frac_df = pd.read_pickle("balance_train_128000.pkl")
test_df = pd.read_pickle("test_public_tweets.pkl")
train_df, _, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)

#%%
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
# build analyzers (bag-of-words)
tfidf_500 = TfidfVectorizer(max_features=5000, tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
tfidf_500.fit(train_df['text'])

# standardize name (X, y) 
X_train = tfidf_500.transform(train_df['text'])
y_train = train_df['emotion']

X_test = tfidf_500.transform(test_df['text'])

#%%
## deal with label (string -> one-hot)

from sklearn.preprocessing import LabelEncoder
import keras

y_train = train_df['emotion']
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)


def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)


print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
#%%
model = load_model('model_balance_train_128000_dp05.h5')

#%%
y_pred = model.predict(X_test)
for i in range(y_pred.shape[0]):
    y_pred[i,np.argmax(y_pred[i,:])] = 1
    y_pred[i,0:np.argmax(y_pred[i,:])] = 0
    y_pred[i,np.argmax(y_pred[i,:])+1:] = 0
y_pred = list(label_decode(label_encoder, y_pred))
#%%
df_submit = pd.DataFrame()
df_submit['id'] = test_df.id
df_submit['emotion'] = y_pred
df_submit.to_csv('submit/submit_model_balance_train_128000_dp05.csv', index=False)
