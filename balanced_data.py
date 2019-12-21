# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:04:49 2019

@author: menghan
"""

# Create a balanced training dataset

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
## load a pickle file
#train_df = pd.read_pickle("train_tweets.pkl")
filename = "train_tweets.pkl"
print('Using ' + filename)
frac_df = pd.read_pickle(filename)
train_df, test_df, _, _ = train_test_split(frac_df, frac_df['emotion'], test_size=0.2, random_state=0)

# the histogram of the data
labels = train_df['emotion'].unique()
post_total = len(train_df)
df1 = train_df.groupby(['emotion']).count()['text']
#df1 = df1.apply(lambda x: round(x,3))

#plot
fig, ax = plt.subplots(figsize=(5,3))
plt.bar(df1.index,df1.values)

#arrange
plt.ylabel('% of instances')
plt.xlabel('Emotion')
plt.title('Emotion distribution')
plt.grid(True)
plt.show()

#%%
probs = {'joy':0.355*2,
         'trust':0.141*2,
         'anticipation':0.171*2,
         'sadness':0.133*2,
         'disgust':0.096*2,
         'fear':0.044*2,
         'surprise':0.033*2,
         'anger':0.027}
count = {'joy':0,
         'trust':0,
         'anticipation':0,
         'sadness':0,
         'disgust':0,
         'fear':0,
         'surprise':0,
         'anger':0}
mask = []


n_sample = 40000
#
for i in range(train_df.shape[0]):
    condition = (np.random.rand(1)[0]>probs[train_df.emotion[i]]) and (count[train_df.emotion[i]]<n_sample)
    if condition:
        mask.append(i)
        count[train_df.emotion[i]] += 1
    if all([x==n_sample for x in list(count.values())]):
        break
mask = np.array(mask)
balance_df = train_df.iloc[mask]

balance_df.to_pickle('balance_train_' + str(n_sample) + '.pkl')
#%%   
## the histogram of the data
#labels = balance_df['emotion'].unique()
#post_total = len(balance_df)
#df1 = balance_df.groupby(['emotion']).count()['text']
#df1 = df1.apply(lambda x: round(x/post_total,3))
#
##plot
#fig, ax = plt.subplots(figsize=(5,3))
#plt.bar(df1.index,df1.values)
#
##arrange
#plt.ylabel('% of instances')
#plt.xlabel('Emotion')
#plt.title('Emotion distribution')
#plt.grid(True)
#plt.show()