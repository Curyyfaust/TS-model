
"""
Created on Tue Feb  6 11:15:15 2024

@author: Yipu
"""

import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataloader
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv,GCNConv
import matplotlib.pyplot as plt

from sklearn import metrics
import numpy as np




def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            #rint(batch.y.size())
            y = batch.y.view(out.size())
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
def Metrics(y_pred,y_true):
    mse=metrics.mean_squared_error(y_true, y_pred)
    mae=metrics.mean_absolute_error(y_true, y_pred)
    
    print(mse)
    print(mae)
    
    return mse,mae
    
def pred(model,test_loader):
    out_list=[]
    true_list=[]
    for batch in test_loader:
        out=model(batch)
        #print(out.shape)
        true=batch.y.view(out.size())
        out_list.append(out.detach().numpy())
        true_list.append(true.detach().numpy())
    
    return np.vstack(out_list),np.vstack(true_list)

def inverse_transform(x):
    
    result=[]
    
    for i in range(x.shape[0]):
        #print(x[0].shape)
        y=x[i].reshape(x[i].shape[0],-1,1)
        y_inverse=[x[0][0] for x in y]
        result.append(y_inverse)
    
    return np.vstack(result)

    