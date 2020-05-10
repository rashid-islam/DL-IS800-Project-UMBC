import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# Collaborative Filtering
# use embedding to build a simple recommendation system
# Source:
# 1. Collaborative filtering, https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# 2. https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
# training neural network based collaborative filtering
# neural network model (NCF)
class neuralCollabFilter(nn.Module):
    def __init__(self, num_users, num_likes, embed_size, num_hidden, d_out, output_size):
        super(neuralCollabFilter, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_size)
        self.like_emb = nn.Embedding(num_likes,embed_size)
        self.fc1 = nn.Linear(embed_size*2, num_hidden[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.relu3 = nn.ReLU()
        self.outLayer = nn.Linear(num_hidden[2], output_size)
        self.drop = nn.Dropout(d_out)
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.like_emb(v)
        out = torch.cat([U,V], dim=1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.outLayer(out)
        return out
#%%
# Autoencoder based collaborative filtering
# Source:
        # 1. https://github.com/SudharshanShanmugasundaram/Movie-Recommendation-System-using-AutoEncoders/blob/master/autoEncoders.py
        # 2. https://github.com/NVIDIA/DeepRecommender/blob/master/reco_encoder/model/model.py
class deepAutoencoder(nn.Module):
    def __init__(self, items, embed_size, num_hidden, d_out):
        super(deepAutoencoder, self).__init__()
        self.encode1 = nn.Linear(items, num_hidden[0])
        self.encode2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.encode3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.encode4 = nn.Linear(num_hidden[2], embed_size)
        self.decode1 = nn.Linear(embed_size,num_hidden[2])
        self.decode2 = nn.Linear(num_hidden[2], num_hidden[1])
        self.decode3 = nn.Linear(num_hidden[1], num_hidden[0])
        self.decode4 = nn.Linear(num_hidden[0],items)
        self.selu = nn.SELU()
        self.drop = nn.Dropout(d_out)
    def forward(self, x):
        out = self.selu(self.encode1(x))
        out = self.selu(self.encode2(out))
        out = self.selu(self.encode3(out))
        out = self.selu(self.encode4(out))
        out = self.selu(self.decode1(self.drop(out)))
        out = self.selu(self.decode2(out))
        out = self.selu(self.decode3(out))
        out = self.decode4(out)
        return out

#%%
# Our proposed deep autoencoder-informed neural collaborative filtering
class autoencoderInformedNCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size, num_hidden, d_out, output_size):
        super(autoencoderInformedNCF, self).__init__()
        # user-based autoencoder
        self.u_encode1 = nn.Linear(num_items, num_hidden[0])
        self.u_encode2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.u_encode3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.u_encode4 = nn.Linear(num_hidden[2], embed_size)
        self.u_decode1 = nn.Linear(embed_size,num_hidden[2])
        self.u_decode2 = nn.Linear(num_hidden[2], num_hidden[1])
        self.u_decode3 = nn.Linear(num_hidden[1], num_hidden[0])
        self.u_decode4 = nn.Linear(num_hidden[0],num_items)
        
        # item-based autoencoder
        self.i_encode1 = nn.Linear(num_users, num_hidden[0])
        self.i_encode2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.i_encode3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.i_encode4 = nn.Linear(num_hidden[2], embed_size)
        self.i_decode1 = nn.Linear(embed_size,num_hidden[2])
        self.i_decode2 = nn.Linear(num_hidden[2], num_hidden[1])
        self.i_decode3 = nn.Linear(num_hidden[1], num_hidden[0])
        self.i_decode4 = nn.Linear(num_hidden[0],num_users)
        
        # neural collaborative filtering
        self.fc1 = nn.Linear(embed_size*2, num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.fc3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.outLayer = nn.Linear(num_hidden[2], output_size)
        
        # utilities
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.drop = nn.Dropout(d_out)
    
    def forward(self, u, i):
        # forward pass of user-based autoencoder
        uae_out = self.selu(self.u_encode1(u))
        uae_out = self.selu(self.u_encode2(uae_out))
        uae_out = self.selu(self.u_encode3(uae_out))
        U = self.selu(self.u_encode4(uae_out))
        uae_out = self.selu(self.u_decode1(self.drop(U)))
        uae_out = self.selu(self.u_decode2(uae_out))
        uae_out = self.selu(self.u_decode3(uae_out))
        uae_out = self.u_decode4(uae_out)
        
        # forward pass of item-based autoencoder
        iae_out = self.selu(self.i_encode1(i))
        iae_out = self.selu(self.i_encode2(iae_out))
        iae_out = self.selu(self.i_encode3(iae_out))
        V = self.selu(self.i_encode4(iae_out))
        iae_out = self.selu(self.i_decode1(self.drop(V)))
        iae_out = self.selu(self.i_decode2(iae_out))
        iae_out = self.selu(self.i_decode3(iae_out))
        iae_out = self.i_decode4(iae_out)
        
        # forward pass of ncf
        ncf_out = torch.cat([U,V], dim=1)
        ncf_out = self.relu(self.fc1(self.drop(ncf_out)))
        ncf_out = self.relu(self.fc2(ncf_out))
        ncf_out = self.relu(self.fc3(ncf_out))
        ncf_out = self.outLayer(ncf_out)
        
        return uae_out, iae_out, ncf_out
