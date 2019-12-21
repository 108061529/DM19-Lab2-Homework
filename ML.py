# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:27:57 2019

@author: menghan
"""

import pickle
import pandas as pd
dim = 300
model_type = 'google'
with open('vector/train_' + model_type + '_' + str(dim) + '.pkl', 'rb') as file:
    train = pickle.load(file)
    
with open('vector/test_' + model_type + '_' + str(dim) + '.pkl', 'rb') as file:
    test = pickle.load(file)
    
## load a pickle file
train_df = pd.read_pickle("train_df.pkl")
test_df = pd.read_pickle("test_df.pkl")
#%%
import numpy as np

def getSentences(vector_sent, dim):
    sentences = np.zeros((len(vector_sent), dim))
    for i in range(len(vector_sent)):
        sentences[i,:] = np.mean(vector_sent[i], axis=0)#######################
    return sentences

train_sent = getSentences(train, dim)
test_sent = getSentences(test, dim)


#%%
X_train = train_sent
X_test = test_sent
y_train = train_df['emotion']
y_test = test_df['emotion']

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore", category=Warning)

classifier = 'svm'

best_score = 0
best_UAR = 0
scores_in_a_pc = []
UARs_in_a_pc = []
complexities = [10]

for pc in range(10, 110, 10):
    
    for comp in complexities:
        print('Complexity = ' + str(comp))
        
        if classifier=='svm':           
            clf_fc = SVC(gamma='scale',C = comp)
#                clf_fc = LinearSVC(random_state=0)
        elif classifier=='rf':
            clf_fc = RandomForestClassifier(criterion='gini', n_estimators=100)
        elif classifier=='mnb':
            clf_fc = MultinomialNB()

        pipe_clf = make_pipeline(SelectPercentile(f_classif, percentile=pc),
                                clf_fc)
        
        pipe_clf.fit(X_train, y_train)
        print('Now Selection Percentile is : ', pc)
        score = pipe_clf.score(X_test, y_test)

        scores_in_a_pc.append(score)
        if score>best_score:
            best_score = score
            which_pc = pc
        print('score = ', round(score,4))
print('Best score = ' + str(round(best_score,4)) + ' at ' + str(which_pc) + ' percent')
