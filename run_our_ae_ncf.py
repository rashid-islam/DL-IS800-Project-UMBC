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

from neural_models import autoencoderInformedNCF

# This script runs NCF model to explicit feedback recommendations
#%%The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% pre-training NCF model with user-page pairs
def train_ncf(model,df_train,train_uae,train_iae, epochs, lr, batch_size, unsqueeze=False):
    criterion_ae = nn.MSELoss(reduction='sum') # mean will be computed later
    criterion_ncf = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            train_user_input, train_item_input, train_ratings = get_cf_data(data_batch,device)
            uae_batch = Variable(train_uae[train_user_input].float()).to(device)
            iae_batch = Variable(train_iae[train_item_input].float()).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
                
            optimizer.zero_grad()                
            uae_out, iae_out, y_hat = model(uae_batch, iae_batch)
            
            # uae loss
            mask_uae = uae_batch != 0
            num_ratings = torch.sum(mask_uae.float())
            loss_uae = criterion_ae(uae_out * mask_uae.float(), uae_batch)
            loss_uae = loss_uae / num_ratings # average loss
            
            # iae loss
            mask_iae = iae_batch != 0
            num_ratings = torch.sum(mask_iae.float())
            loss_iae = criterion_ae(iae_out * mask_iae.float(), iae_batch)
            loss_iae = loss_iae / num_ratings # average loss
            
            # ncf loss
            loss_ncf = criterion_ncf(y_hat, train_ratings)
            
            # total loss
            loss = 0.9*loss_ncf + (1-0.9)*(loss_uae + loss_iae)
            loss.backward()
            optimizer.step()
#%% evaluations
def test_model(model,df_val,train_uae,train_iae,unsqueeze=False):
    model.eval()
    test_user_input, test_item_input, test_ratings = get_cf_data(df_val,device)
    uae_test = Variable(train_uae[test_user_input].float()).to(device)
    iae_test = Variable(train_iae[test_item_input].float()).to(device)
    if unsqueeze:
        test_ratings = test_ratings.unsqueeze(1)
        
    _,_, y_hat = model(uae_test, iae_test)
    
    test_ratings = test_ratings.cpu().detach().numpy().reshape((-1,))
    y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
    
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
def evaluate_ranking(model,df_val,train_uae,train_iae,top_K,random_samples, num_items, interactions):
    model.eval()
    avg_HR = np.zeros((len(df_val),top_K))
    avg_NDCG = np.zeros((len(df_val),top_K))
    
    for i in range(len(df_val)):
        test_user_input, test_item_input = get_test_instances_with_random_samples(df_val[i], random_samples,num_items,interactions,device)
        uae_test = Variable(train_uae[test_user_input].float()).to(device)
        iae_test = Variable(train_iae[test_item_input].float()).to(device)
        
        _,_, y_hat = model(uae_test, iae_test)
        
        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
        
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

train_data_uae =  convert_ae(train_ratings, numUsers, numItems)
train_data_iae = np.transpose(train_data_uae)

train_data_uae = torch.from_numpy(train_data_uae)
train_data_iae = torch.from_numpy(train_data_iae)
#%% set hyperparameters
emb_size = [256, 128, 64]
hidden_layers = [[512, 256, 128],[500, 400, 300],[400, 300, 200],[256, 128, 64],[128, 64, 32]]
output_size = 1
num_epochs = 10
lr = 0.001 #learning_rate = [0.0001, 0.001, 0.01]
d_out = 0.5 # drop_probs = [0.0, 0.5]
batch_size = 256 

random_samples = 100
top_K = 10

#%% hyperparameter selection to choose best NCF model
import sys
sys.stdout=open("results/proposed_ae_ncf_model_results.txt","w")

best_ae_ncf = None
best_out = 1000 # mse is the best_out for explicit feedback problem

for vec in emb_size:
  for layers in hidden_layers:         
    print(f"hyper-parameter configurations:")
    print('vector size: ', vec, 'hidden layers: ', layers, 'learning rate: ', lr, 'Drop out prob: ', d_out)
    
    ae_ncf = autoencoderInformedNCF(numUsers, numItems, vec, layers, d_out, output_size).to(device)
    train_ncf(ae_ncf,train_ratings,train_data_uae,train_data_iae, num_epochs, lr, batch_size, unsqueeze=True)
    # check on validation set:
    mse = test_model(ae_ncf,val_ratings, train_data_uae, train_data_iae, unsqueeze=True)
    #avg_HR,avg_NDCG = evaluate_ranking(ae_ncf,val_ratings.values,train_data_uae, train_data_iae,top_K,random_samples, numItems,interactions)
    # update the best model
    if mse < best_out:
      best_out = mse
      best_ae_ncf = ae_ncf
      print('\n')
      print(f"Current best model:")
      print('vector size: ', vec, 'hidden layers: ', layers, 'learning rate: ', lr, 'Drop out prob: ', d_out)
          
torch.save(best_ae_ncf.state_dict(), "trained-models/best_ae_ncf")
#%% evaluate on the test set
print('\n')
print(f"Evaluation on the test dataset:")
test_model(best_ae_ncf,test_ratings,train_data_uae, train_data_iae, unsqueeze=True)
avg_HR, avg_NDCG = evaluate_ranking(best_ae_ncf,test_ratings.values,train_data_uae, train_data_iae,top_K,random_samples, numItems,interactions)

np.savetxt('results/avg_HR_ae_ncf.txt',avg_HR)
np.savetxt('results/avg_NDCG_ae_ncf.txt',avg_NDCG)
