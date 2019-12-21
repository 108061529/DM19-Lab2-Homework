# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:09:51 2019

@author: menghan
"""

import pickle
import pandas as pd


## load a pickle file
filename = "test_public_tweets.pkl"
frac_df = pd.read_pickle(filename)
print('Postag and stem ' + filename)

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords
import re

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in nltk.word_tokenize(text)] ###########
def tokenizer(text):
    return text.split()

nltk.download('stopwords')
stop = stopwords.words('english')
stop = [tokenizer_porter(item)[0] for item in stop]
stop.extend(['becau'])

df4 = frac_df.copy()   
pos = []
for i in range(frac_df.shape[0]):
    sent = frac_df.text.iloc[i]
    sent = sent.replace("@", "")
    sent = sent.replace(".", "")
    sent = re.sub(r'[^\w\s\r]',' ',sent)
#    print(len(word_tokenize(sent)))
    stemmed = tokenizer_porter(sent)
    tags = pos_tag([word for word in nltk.word_tokenize(sent)])###########
    stem_with_pos = []
    for j in range(len(tags)):
        stem_with_pos.append(stemmed[j] + '_' + tags[j][1])
    pos.append( ' '.join(stem_with_pos))
#    df4.text.iloc[i] = ' '.join(stem_with_pos)
df4.text = pos    
file = open(filename.split('.')[0] + '_stemmed_without_punc_postag_nltk_token.pkl', 'wb')
pickle.dump(df4, file)
file.close()
print('Saved') 












