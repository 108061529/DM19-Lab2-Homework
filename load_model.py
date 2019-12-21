# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 02:17:37 2019

@author: menghan
"""
from keras.models import load_model

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
## load a pickle file
frac_df = pd.read_pickle("balance_train_128000.pkl")
#frac_df = train_all_df.sample(frac=0.144,random_state=0)
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)

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
y_test = test_df['emotion']

#%%
## deal with label (string -> one-hot)

from sklearn.preprocessing import LabelEncoder
import keras

y_train = train_df['emotion']
y_test = test_df['emotion']
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)
#%%
model = load_model('lstm_model_balance_train_128000.h5')
#%%
y_pred = model.predict(X_test)
for i in range(y_pred.shape[0]):
    y_pred[i,np.argmax(y_pred[i,:])] = 1
    y_pred[i,0:np.argmax(y_pred[i,:])] = 0
    y_pred[i,np.argmax(y_pred[i,:])+1:] = 0

# precision, recall, f1-score,
y_test = list(label_decode(label_encoder, y_test))
y_pred = list(label_decode(label_encoder, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_pred))
