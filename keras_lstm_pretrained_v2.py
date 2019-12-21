# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:24:54 2019

@author: menghan
"""

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input
from keras.layers import  Embedding, Softmax
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import Model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
import pandas as pd


BASE_DIR = ''
GLOVE_DIR = 'C:/Users/menghan/Desktop/glove.6B'
MAX_SEQUENCE_LENGTH = 30
maxlen = MAX_SEQUENCE_LENGTH
MAX_NUM_WORDS = 20000
max_features = MAX_NUM_WORDS
EMBEDDING_DIM = 100
batch_size = 512

embeddings_index = {}
with open('C:/Users/menghan/Desktop/glove.6B/glove.6B.100d.txt',encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

#%%
filename = "train_tweets.pkl"
print('Using ' + filename)
frac_df = pd.read_pickle(filename)
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)
#%%
tokenizer_keras = Tokenizer(filters='"#%&()*+,-./:;<=>@[\]^`{|}~', num_words = max_features)
tokenizer_keras.fit_on_texts(train_df['text'])
#%%
word_index = tokenizer_keras.word_index
x_train = tokenizer_keras.texts_to_sequences(train_df['text'])
x_test = tokenizer_keras.texts_to_sequences(test_df['text'])
#%%
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

# O check

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)
#%%
padded_tokens = sequence.pad_sequences(x_train,  maxlen=maxlen, padding="post")
x_train = list(padded_tokens)
padded_tokens = sequence.pad_sequences(x_test,  maxlen=maxlen, padding="post")
x_test = list(padded_tokens)
#%%
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
#%%
print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#%%
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
#%%
print('Build model...')
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32,activation="relu"))
model.add(Dense(8, activation='relu'))
model.add(Softmax())

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
#%%
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('logs/balance_train_lstm_glove100.csv')
print('Train...')
history = model.fit(x_train, y_train,
                    callbacks=[csv_logger],
                    batch_size=batch_size,
                    epochs=6,
                    validation_data=(x_test, y_test))
#score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)
#%%
y_pred = model.predict(x_test)
for i in range(y_pred.shape[0]):
    y_pred[i,np.argmax(y_pred[i,:])] = 1
    y_pred[i,0:np.argmax(y_pred[i,:])] = 0
    y_pred[i,np.argmax(y_pred[i,:])+1:] = 0
#%%
y_test = list(label_decode(label_encoder, y_test))
y_pred = list(label_decode(label_encoder, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_pred))

#%%
model.save('models/lstm_model_balance_train_tweets_glove100.h5')
#%%
#import matplotlib.pyplot as plt
#training_log = pd.read_csv('logs/balance_train_lstm_glove_twitter_100.csv')
#
#plt.figure(0)
#plt.title('Accuracy per epoch')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.plot(training_log.acc, label = 'Train accuracy')
#plt.plot(training_log.val_acc, label = 'Test accuracy')
#plt.legend()
#
#plt.figure(1)
#plt.title('Loss curve')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.plot(training_log.loss, label = 'Train Loss')
#plt.plot(training_log.val_loss, label = 'Test Loss')
#plt.legend()
print('training finish')