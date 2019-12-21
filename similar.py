# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:06:24 2019

@author: menghan
"""
import numpy as np
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

# happy for joy
# keep joy
# sad for sadness
# angry for anger
# scary for fear
# add ew
targets = ['happy',
           'joy',
         'trust',
         'anticipation',
         'sad',
         'disgust',
         'scary',
         'surprise',
         'angry',
         'ew']
vocab = []
top_n = 100
for t in targets:
    tt=model.most_similar(t, topn=top_n)
    vocab.extend([tt[x][0] for x in range(top_n)])

import pickle
file = open('balance_train_64000_vocab_'+ str(top_n) +'_' + model_type + '_' + str(dim) + '.pkl', 'wb')
pickle.dump(vocab, file)
file.close()