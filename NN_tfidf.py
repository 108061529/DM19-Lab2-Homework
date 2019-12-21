# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:14:06 2019

@author: menghan
"""

import pickle
import pandas as pd
import numpy as np
## load a pickle file
train_df = pd.read_pickle("train_df.pkl")
test_df = pd.read_pickle("test_df.pkl")

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
from keras.layers import Dropout

# I/O check
input_shape = X_train.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)

from keras.models import Model
from keras.layers import Input, Dense, Multiply
from keras.layers import ReLU, Softmax

# input layer
model_input = Input(shape=(input_shape, ))  # 500
X = model_input

# 1st hidden layer
X_W1 = Dense(units=128)(X)  # 64
H1_D = Dropout(0.6)(X_W1)
H1 = ReLU()(H1_D)
#H1 = ReLU()(X_W1)
#H1_D = Dropout(0.6)(H1)

# 2nd hidden layer
H1_W2 = Dense(units=64)(H1)  # 64
H2_D = Dropout(0.6)(H1_W2)
H2 = ReLU()(H2_D)
#H2 = ReLU()(H1_W2)
#H2_D = Dropout(0.6)(H2)

# output layer
H2_W3 = Dense(units=output_shape)(H2)  # 4
H3 = Softmax()(H2_W3)

model_output = H3

# create model
model = Model(inputs=[model_input], outputs=[model_output])

# loss function & optimizer
optim = keras.optimizers.adam(learning_rate = 0.001) 
model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# show model construction
model.summary()

#%%
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('logs/training_log_sentence_vector.csv')

# training setting
epochs = 300
batch_size = 1024

# training!
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[csv_logger],
                    validation_data = (X_test, y_test))
#%%
import matplotlib.pyplot as plt
training_log = pd.read_csv('logs/training_log_sentence_vector.csv')

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
