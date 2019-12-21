# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:01:02 2019

@author: menghan
"""

import json

file = open(r'D:\DMLAB\dm19-lab2-nthu\tweets_DM.json', 'r', encoding='utf-8')
all_ = []
for line in file.readlines():
    dic = json.loads(line)
    all_.append(dic)

import pandas as pd
#emo = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\emotion.csv')
#ident = pd.read_csv(r'D:\DMLAB\dm19-lab2-nthu\data_identification.csv')
#%%
#df_all = pd.DataFrame()
ids = []
text = []
hashtags = []
score = []
date = []
index = []
#type_ = []
for i in range(len(all_)):
    ids.append(all_[i]['_source']['tweet']['tweet_id'])
    hashtags.append(all_[i]['_source']['tweet']['hashtags'])
    text.append(all_[i]['_source']['tweet']['text'])
    score.append(all_[i]['_score'])
    date.append(all_[i]['_crawldate'])
    index.append(all_[i]['_index'])
#df_all['id'] = ids
#df_all['hashtags'] = hashtags
#df_all['text'] = text
#df_all['score'] = score
#df_all['date'] = date
#df_all['index'] = index

#df_all.to_csv('all_tweets.csv')
#%%
all_dict = dict()
for i in range(len(all_)):
    all_dict.update({ids[i]: [hashtags[i], text[i], score[i], date[i], index[i]]})
#%%
import pickle
file = open('all_tweets.pkl', 'wb')
pickle.dump(all_dict, file)
file.close()     
    
    