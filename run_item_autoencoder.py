
import numpy as np
import pandas as pd
import heapq # for retrieval topK

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from utilities import load_interactions, get_cf_data, get_test_instances_with_random_samples, getHitRatio, getNDCG, convert_ae

from neural_models import deepAutoencoder

# This script runs NCF model to explicit feedback recommendations
#%%The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% train autoencoder model with user-item rating matrix
def train_ae(model,train_data, epochs, lr, batch_size):
    criterion = nn.MSELoss(reduction='sum') # mean will be computed later
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        for batch_i in range(0,np.int64(np.floor(len(train_data)/batch_size))*batch_size,batch_size):
            data_batch = train_data[batch_i:(batch_i+batch_size)]
            data_batch = Variable(data_batch.float()).to(device)
            
            optimizer.zero_grad()
            output = model(data_batch)
            # optimize only for rated items
            mask = data_batch != 0
            num_ratings = torch.sum(mask.float())
            loss = criterion(output * mask.float(), data_batch)
            loss = loss / num_ratings # average loss
            
            loss.backward()
            optimizer.step()
#%% evaluations
def test_model(model,test_data,test_ratings):
    model.eval()
    
    test_data = Variable(test_data.float()).to(device)
    output = model(test_data)
    
    output = np.transpose(output.cpu().detach().numpy())
    
    y_hat = np.zeros((len(test_ratings)))
    for i in range(len(test_ratings)):
        y_hat[i] = output[test_ratings['userId'][i],test_ratings['movieId'][i]]
        
    test_ratings = test_ratings['rating'].values   
    mae = mean_absolute_error(test_ratings, y_hat)
    print(f"mean_absolute_error: {mae: .3f}")
    mse = mean_squared_error(test_ratings, y_hat)
    print(f"mean_squared_error: {mse: .3f}")
    rmse = np.sqrt(mse)
    print(f"root_mean_squared_error: {rmse: .3f}")
    r2 = r2_score(test_ratings, y_hat) # higher the better
    print(f"r2 score: {r2: .3f}")
    return mse

#%% model evaluation: hit rate and NDCG
def evaluate_ranking(model,test_data, test_ratings,top_K,random_samples, num_items, interactions):
    model.eval()
    avg_HR = np.zeros((len(test_ratings),top_K))
    avg_NDCG = np.zeros((len(test_ratings),top_K))
    
    test_data = Variable(test_data.float()).to(device)
    output = model(test_data)
    output = np.transpose(output.cpu().detach().numpy())
    
    for i in range(len(test_ratings)):
        test_user_input, test_item_input = get_test_instances_with_random_samples(test_ratings[i], random_samples,num_items,interactions,device)

        test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
        test_user_input = test_user_input.cpu().detach().numpy().reshape((-1,))
        y_hat = output[test_user_input,test_item_input]
        
        map_item_score = {}
        for j in range(len(y_hat)):
            map_item_score[test_item_input[j]] = y_hat[j]
        for k in range(top_K):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = test_item_input[0]
            avg_HR[i,k] = getHitRatio(ranklist, gtItem)
            avg_NDCG[i,k] = getNDCG(ranklist, gtItem)
    avg_HR = np.mean(avg_HR, axis = 0)
    avg_NDCG = np.mean(avg_NDCG, axis = 0)
    print(f"avg HR")
    print(avg_HR)
    print(f"avg NDCG")
    print(avg_NDCG)
    return avg_HR, avg_NDCG
    


#%% run NCF model on the movieLens dataset
train_ratings = pd.read_csv("data/train_ratings.csv")
val_ratings = pd.read_csv("data/val_ratings.csv")
test_ratings = pd.read_csv("data/test_ratings.csv")
interactions=load_interactions('data/interactions.txt')

numUsers = len(train_ratings.userId.unique())
numItems = len(train_ratings.movieId.unique())

train_data_ae =  np.transpose(convert_ae(train_ratings, numUsers, numItems))
#train_data_iae = np.transpose(train_data_uae)
val_data_ae =  np.transpose(convert_ae(val_ratings, numUsers, numItems))
test_data_ae =  np.transpose(convert_ae(test_ratings, numUsers, numItems))

train_data_ae = torch.from_numpy(train_data_ae)
val_data_ae = torch.from_numpy(val_data_ae)
test_data_ae = torch.from_numpy(test_data_ae)

#%% set hyperparameters
emb_size = [32, 64, 128, 256]
hidden_layers = [[512, 256, 128],[256, 128, 64],[128, 64, 32], [64, 32, 16]]
output_size = 1
num_epochs = 10
learning_rate = [0.0001, 0.001, 0.01]
drop_probs = [0.0, 0.1, 0.25, 0.5]
batch_size = 256 

random_samples = 100
top_K = 10


#%% hyperparameter selection to choose best NCF model
import sys
sys.stdout=open("results/item_based_ae_results.txt","w")

best_iae = None
best_out = 1000 # mse is the best_out for explicit feedback problem

for vec in emb_size:
  for layers in hidden_layers:
    for lr in learning_rate: 
        for d_out in drop_probs:          
            print(f"hyper-parameter configurations:")
            print('vector size: ', vec, 'hidden layers: ', layers, 'learning rate: ', lr, 'Drop out prob: ', d_out)
            
            ae = deepAutoencoder(numUsers, vec, layers, d_out).to(device)
            train_ae(ae,train_data_ae, num_epochs, lr, batch_size)

            # check on validation set:
            mse = test_model(ae,train_data_ae,val_ratings)
            avg_HR,avg_NDCG = evaluate_ranking(ae,train_data_ae,val_ratings.values,top_K,random_samples, numItems,interactions)
            # update the best model
            if mse < best_out:
              best_out = mse
              best_iae = ae
              print('\n')
              print(f"Current best model:")
              print('vector size: ', vec, 'hidden layers: ', layers, 'learning rate: ', lr, 'Drop out prob: ', d_out)
          
torch.save(best_iae.state_dict(), "trained-models/best_iae")
#%% evaluate on the test set
print('\n')
print(f"Evaluation on the test dataset:")
test_model(best_iae,train_data_ae,test_ratings)
avg_HR, avg_NDCG = evaluate_ranking(best_iae,train_data_ae,test_ratings.values,top_K,random_samples, numItems,interactions)

np.savetxt('results/avg_HR_iae.txt',avg_HR)
np.savetxt('results/avg_NDCG_iae.txt',avg_NDCG)
