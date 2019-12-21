# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:56:31 2019

@author: menghan
"""

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.layers import ReLU, Softmax
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

maxlen = 30 # arbitrary
batch_size = 512
max_features = 20000
#%%
filename = "train_tweets.pkl"
print('Using ' + filename)
frac_df = pd.read_pickle(filename)
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)
#%%
tokenizer_keras = Tokenizer(filters='"#%&()*+,-./:;<=>@[\]^`{|}~', num_words = max_features)
tokenizer_keras.fit_on_texts(train_df['text'])
#%%
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
#%%
#pad sequences
x_train = sequence.pad_sequences(x_train,  maxlen=maxlen, padding="post")
x_test = sequence.pad_sequences(x_test,  maxlen=maxlen, padding="post")

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
#%%
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(8, activation='relu'))
model.add(Softmax())

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('logs/qq.csv')
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
#model.save('models/lstm_model_balance_train_256000.h5')
#%%
import matplotlib.pyplot as plt
training_log = pd.read_csv('logs/qq.csv')

plt.figure(0)
plt.title('Accuracy per epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(training_log.accuracy, label = 'Train accuracy')
plt.plot(training_log.val_accuracy, label = 'Test accuracy')
plt.legend()

plt.figure(1)
plt.title('Loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(training_log.loss, label = 'Train Loss')
plt.plot(training_log.val_loss, label = 'Test Loss')
plt.legend()
print('training finish')