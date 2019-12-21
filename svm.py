# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:02:44 2019

@author: menghan
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
classifier='svm'
## load a pickle file
filename = "balance_train_256000.pkl"
print('Using ' + filename)
frac_df = pd.read_pickle(filename)
#frac_df = train_all_df.sample(frac=0.144,random_state=0)
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)

#%%
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
## build analyzers (bag-of-words)
#tfidf_500 = TfidfVectorizer(ngram_range=(1,1), lowercase=False, tokenizer=nltk.word_tokenize) 
##tfidf_500 = TfidfVectorizer(vocabulary = list(np.unique(vocab)), tokenizer=nltk.word_tokenize) 
#
## apply analyzer to training data
#tfidf_500.fit(train_df['text'])
#
## standardize name (X, y) 
#X_train = tfidf_500.transform(train_df['text'])
#y_train = train_df['emotion']
#
#X_test = tfidf_500.transform(test_df['text'])
#y_test = test_df['emotion']
#print(X_train.shape)
#%%
def tokenizer(text):
    return text.split()
#%%


best_score = 0
best_UAR = 0
scores_in_a_pc = []
UARs_in_a_pc = []
complexities = [1]
for pc in range(5, 10, 5):
        
        for comp in complexities:
            print('Complexity = ' + str(comp))
            
            if classifier=='svm':           
#                clf_fc = SVC(gamma='scale',C = comp)
                clf_fc = LinearSVC(C = comp, random_state=0)
            elif classifier=='rf':
                clf_fc = RandomForestClassifier(criterion='entropy')
            elif classifier=='mnb':
                clf_fc = MultinomialNB()

            pipe_clf = make_pipeline(TfidfVectorizer(ngram_range=(1,1), lowercase=False, tokenizer=tokenizer),
                                     SelectPercentile(f_classif, percentile=pc),
                                    clf_fc)
            
            pipe_clf.fit(train_df['text'], train_df['emotion'])
            print('Now Selection Percentile is : ', pc)
            
            y_pred = pipe_clf.predict(test_df['text'])

from sklearn.metrics import classification_report
print(classification_report(y_true=test_df['emotion'], y_pred=y_pred))

with open('D:/DMLAB/DMLab2/DM19-Lab2-Homework/models/' +  'lin_svm_balance_train_256000_with_tfidf.pkl', 'wb') as f:
    pickle.dump(pipe_clf, f)
print('Saved')