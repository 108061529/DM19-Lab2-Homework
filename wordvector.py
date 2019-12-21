# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:04:14 2019

@author: menghan
"""

import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api
dim = 25
model_type = 'twitter'
print('Using '+ model_type + ' wv model')
def getVector(tweet, dim):
    cnt = []
    c=0
    vectors = np.zeros((len(tweet), dim))
    for i in range(len(tweet)):
        try:
            vectors[i,:] = model[tweet[i]]
        except:
            c+=1
#            print(tweet[i] + ' not in vocabulary')
    cnt.append(c)
    return vectors
# If you see `SSL: CERTIFICATE_VERIFY_FAILED` error, use this:
#import ssl
#import urllib.request
#ssl._create_default_https_context = ssl._create_unverified_context
#
#model = api.load("glove-twitter-100")
if model_type == 'twitter':
    model = KeyedVectors.load_word2vec_format('D:/DMLAB/glove-twitter-'+str(dim)+'.gz')
    print('load ok')
elif model_type == 'google':
    model = KeyedVectors.load_word2vec_format('D:/DMLAB/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('load ok')

#%%
from sklearn.model_selection import train_test_split
## load a pickle file
frac_df = pd.read_pickle("balance_train_64000.pkl")
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)
#%%
import nltk
import numpy as np
tokenizer = nltk.word_tokenize
training_tokens = list(train_df['text'].apply(lambda x: nltk.word_tokenize(x)))
testing_tokens = list(test_df['text'].apply(lambda x: nltk.word_tokenize(x)))

training_vectors = [getVector(x, dim) for x in training_tokens]
testing_vectors = [getVector(x, dim) for x in testing_tokens]

#%%
import pickle
file = open('vector/balance_train_64000_' + model_type + '_' + str(dim) + '.pkl', 'wb')
pickle.dump(training_vectors, file)
file.close()  

file = open('vector/balance_test_64000_' + model_type + '_' + str(dim) + '.pkl', 'wb')
pickle.dump(testing_vectors, file)
file.close() 











