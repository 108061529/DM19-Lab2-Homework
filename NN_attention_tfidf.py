# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:32:49 2019

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
tfidf_500 = TfidfVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 

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
#%%
from keras.layers import Dropout, Multiply

# I/O check
input_shape = X_train.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ReLU, Softmax

# input layer
model_input = Input(shape=(input_shape, ))  # 500
inputs = model_input



# ATTENTION PART STARTS HERE
attention_probs = Dense(input_shape, activation='softmax', name='attention_vec')(inputs)
attention_mul = Multiply()([inputs, attention_probs])
# ATTENTION PART FINISHES HERE

#attention_mul = Dense(128)(attention_mul)

H1_W2 = Dense(units=128)(attention_mul)  # 64
H2_D = Dropout(0.6)(H1_W2)
H2 = ReLU()(H2_D)

H2_W3 = Dense(units=64)(H2)  # 64
H3_D = Dropout(0.6)(H2_W3)
H3 = ReLU()(H3_D)


output = Dense(output_shape, activation='softmax')(H3)

model = Model(input=[inputs], output=output)

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
epochs = 200
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













