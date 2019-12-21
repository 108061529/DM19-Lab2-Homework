# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:23:33 2019

@author: menghan
"""

import pickle
import pandas as pd

sub = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\sampleSubmission.csv')
with open('all_tweets.pkl', 'rb') as file:
    all_dict = pickle.load(file)
    
ids = []
hashtags = []
text = []
score = []
date = []
index = []

for i in range(sub.shape[0]):
    ids.append(sub.id[i])
    hashtags.append(all_dict[sub.id[i]][0])
    text.append(all_dict[sub.id[i]][1])
    score.append(all_dict[sub.id[i]][2])
    date.append(all_dict[sub.id[i]][3])
    index.append(all_dict[sub.id[i]][4])
df = pd.DataFrame()
df['id'] = ids
df['hashtags'] = hashtags
df['text'] = text
df['score'] = score
df['date'] = date
df['index'] = index
df.to_pickle('test_public_tweets.pkl')
#%%
emo_df = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\emotion.csv')
for i in range(emo_df.shape[0]):
    if ids[0]==emo_df.tweet_id[i]:
        print('yabai')