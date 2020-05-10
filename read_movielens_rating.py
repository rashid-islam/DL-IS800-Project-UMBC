import io
import os
import math
import copy
import pickle
import zipfile
from textwrap import wrap
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# def try_download(url, download_path):
#     archive_name = url.split('/')[-1]
#     folder_name, _ = os.path.splitext(archive_name)
#     
#     try:
#         r = urlopen(url)
#     except URLError as e:
#         print('Cannot download the data. Error: %s' % s)
#         return 
# 
#     assert r.status == 200
#     data = r.read()
# 
#     with zipfile.ZipFile(io.BytesIO(data)) as arch:
#         arch.extractall(download_path)
#         
#     print('The archive is extracted into folder: %s' % download_path)
# =============================================================================

def read_data(path):
    files = {}
    for filename in path.glob('*'):
        if filename.suffix == '.csv':
            files[filename.stem] = pd.read_csv(filename)
        elif filename.suffix == '.dat':
            if filename.stem == 'ratings':
                columns = ['userId', 'movieId', 'rating', 'timestamp']
            elif filename.stem == 'users':
                columns = ['userId', 'gender', 'age','occupation', 'zip']
            else:
                columns = ['movieId', 'title', 'genres']
            data = pd.read_csv(filename, sep='::', names=columns, engine='python')
            files[filename.stem] = data
    return files['ratings'], files['movies'], files['users']

#archive_url = f'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
download_path = Path.home() / 'data' / 'movielens'

#try_download(archive_url, download_path)

ratings, movies, users = read_data(download_path / 'ml-1m')

#ratings=ratings.rename(columns={"userId": "user_id", "movieId": "like_id"})
ratings = (pd.concat([ratings['userId']-1,ratings['movieId'],ratings['rating']],axis=1)).reset_index(drop=True)

counts = ratings['movieId'].value_counts()
ratings = (ratings[~ratings['movieId'].isin(counts[counts < 5].index)]).reset_index(drop=True)

#%% data statistics
import sys
sys.stdout=open("data/data_statistics.txt","w")
print(f"Data Statistics for Movielens data set")

print(f"Number of movies: {len(np.unique(ratings['movieId'])): .2f}")
print(f"Number of users: {len(np.unique(ratings['userId'])): .2f}")
print(f"Number of user-item interactions: {len(ratings): .2f}")

#%% user-pages: train-dev-test split
le = LabelEncoder()
columnUnique=list(ratings['movieId'].unique())
le_fitted_concentr = le.fit(columnUnique)
col_values=list(ratings['movieId'].values)
le.classes_
col_valuesUser=le.transform(col_values)
ratings['movieId'] = col_valuesUser

np.random.seed(7)
msk = np.random.rand(len(ratings)) < 0.99
train_ratings = (ratings[msk].copy()).reset_index(drop=True)
test_ratings = (ratings[~msk].copy()).reset_index(drop=True)

msk = np.random.rand(len(train_ratings)) < 0.99
new_ratings = train_ratings
train_ratings = (new_ratings[msk].copy()).reset_index(drop=True)
val_ratings = (new_ratings[~msk].copy()).reset_index(drop=True)

# =============================================================================
# col_values=list(test_ratings['movieId'].values) 
# col_valuesUser=le.transform(col_values)
# test_ratings['movieId'] = col_valuesUser
# 
# col_values=list(val_ratings['movieId'].values) 
# col_valuesUser=le.transform(col_values)
# val_ratings['movieId'] = col_valuesUser
# =============================================================================

train_ratings.to_csv('data/train_ratings.csv',index=False)
val_ratings.to_csv('data/val_ratings.csv',index=False)
test_ratings.to_csv('data/test_ratings.csv',index=False)

#%% user-item interactions formation
numUsers = len(np.unique(ratings['userId']))
interactions = list()

for i in range(numUsers):
    interactions.append(list(ratings['movieId'][ratings['userId']==i]))
    
f=open('data/interactions.txt','w')
for i in range(len(interactions)):
    for j in interactions[i]:
        f.write('%d' %(j))
        f.write(' ')
    f.write('\n')
f.close()





