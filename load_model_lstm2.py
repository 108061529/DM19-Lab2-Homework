# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:14:34 2019

@author: menghan
"""

from keras.models import load_model

import pickle
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

maxlen = 30 # lstm
#maxlen = 100 # cnn lstm
batch_size = 512
max_features = 20000
## load a pickle file
filename = "train_tweets.pkl"
model_name = 'lstm_model_train_tweets_maxfeature_40000.h5'
print('Using ' + filename)
frac_df = pd.read_pickle(filename)
test_df = pd.read_pickle("test_public_tweets.pkl")
train_df, _, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)
#%%
tokenizer_keras = Tokenizer(filters='"#%&()*+,-./:;<=>@[\]^`{|}~', num_words = max_features)
tokenizer_keras.fit_on_texts(train_df['text'])
#%%
x_test = tokenizer_keras.texts_to_sequences(test_df['text'])
#%%
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

#%%
padded_tokens = sequence.pad_sequences(x_test,  maxlen=maxlen, padding="post")
x_test = list(padded_tokens)
#%%
print('Pad sequences (samples x time)')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_test shape:', x_test.shape)
#%%
model = load_model('models/'+model_name)
#%%
y_pred = model.predict(x_test)
for i in range(y_pred.shape[0]):
    y_pred[i,np.argmax(y_pred[i,:])] = 1
    y_pred[i,0:np.argmax(y_pred[i,:])] = 0
    y_pred[i,np.argmax(y_pred[i,:])+1:] = 0

y_pred = list(label_decode(label_encoder, y_pred))
#%%
df_submit = pd.DataFrame()
df_submit['id'] = test_df.id
df_submit['emotion'] = y_pred
df_submit.to_csv('submit/submit_' + model_name.split('.')[0] +'.csv', index=False)