# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:26:43 2019

@author: menghan
"""
import pickle
import pandas as pd

ident = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\data_identification.csv')
with open('all_tweets.pkl', 'rb') as file:
    all_dict = pickle.load(file)
emo_df = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\emotion.csv')
#%%
ids = []
hashtags = []
text = []
score = []
date = []
index = []

for i in range(ident.shape[0]):
    if ident.identification[i]=='train':
        ids.append(ident.tweet_id[i])
        hashtags.append(all_dict[ident.tweet_id[i]][0])
        text.append(all_dict[ident.tweet_id[i]][1])
        score.append(all_dict[ident.tweet_id[i]][2])
        date.append(all_dict[ident.tweet_id[i]][3])
        index.append(all_dict[ident.tweet_id[i]][4])
df = pd.DataFrame()
df['id'] = ids
df['hashtags'] = hashtags
df['text'] = text
df['score'] = score
df['date'] = date
df['index'] = index
#%%
emo_dict = dict()
for i in range(emo_df.shape[0]):
    emo_dict.update({emo_df.tweet_id[i]: emo_df.emotion[i]})

#%%
emos = []
for i in range(len(df.id)):
    emos.append(emo_dict[df.id[i]])
df['emotion'] = emos
#%%
#df.to_pickle('train_tweets.pkl')
