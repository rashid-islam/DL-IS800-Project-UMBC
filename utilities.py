# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:32:57 2020

@author: islam
"""
import pandas as pd
import numpy as np
from numpy.random import choice
import math

import torch
import torch.nn as nn

#%% Load user-item interactions
def load_interactions(path):
    interactions=list()
    #filename='data/interactions.txt'
    f=open(path,'r')
    for line in f:
        interactionsTemp=list()
        tmp=line.split()
        for item in tmp:
            interactionsTemp.append(int(item)) 
        interactions.append(interactionsTemp)
    f.close()
    return interactions
#%% data formation to run the model
def get_cf_data(data,device):
    return torch.LongTensor(data['userId'].values).to(device), torch.LongTensor(data['movieId'].values).to(device), torch.FloatTensor(data['rating'].values).to(device)

#%% data formation to evaluate ranking
def get_test_instances_with_random_samples(data, random_samples,num_items,interactions,device):
    user_input = np.zeros((random_samples+1))
    item_input = np.zeros((random_samples+1))
    
    # positive instance
    user_input[0] = data[0]
    item_input[0] = data[1]
    i = 1
    # negative instances
    checkList = interactions[data[0]]
    for t in range(random_samples):
        j = np.random.randint(num_items)
        while j in checkList:
            j = np.random.randint(num_items)
        user_input[i] = data[0]
        item_input[i] = j
        i += 1
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device)

#%% performance measures: hit rate and NDCG
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
#%% data conversion for autoencoder based recommender systems
def convert_ae(ratings, num_users, num_items):
    data = np.zeros((num_users,num_items))
    for i in range(len(ratings)):
        data[ratings['userId'][i],ratings['movieId'][i]] = ratings['rating'][i]
    return data


