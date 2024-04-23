# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:42:12 2024

@author: Yipu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from Embed import DataEmbedding



class SDNet(nn.Module):
    #local-global layer
    def __init__(self, feature_size=256, n_heads=4, dropout=0.1,conv_kernel=[4,2],
                 casual_kernel=[6], device='cuda'):
        super(SDNet,self).__init__()
        
        #Conv1d-local
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])
        
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])
        
        
        #Casual Conv-global
        self.casual_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in casual_kernel])
        
        
        
        
        
        # feedforward network
        #self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        #self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size*n_heads)
        self.norm2 = nn.LayerNorm(feature_size*n_heads)

        
        #Other part
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.1)
    
    
    def forward(self, x):
        #B,T,C=x.shape
        B,seq_len ,C=x.shape
        x = x.permute(0, 2, 1)
        
        #multi-head
        mg=torch.tensor([])
        
        #n_heads=4
        for i in range(4):
            xi=x
            #local-feature
            xi=self.conv[0](xi)
            xi=self.conv[1](xi)
            
            zeros = torch.zeros((xi.shape[0], xi.shape[1], xi.shape[2] - 1))
            xi = torch.cat((zeros, xi), dim=-1)
            xi = self.drop(self.act(self.casual_conv[0](xi)))
            
            xi=self.conv_trans[0](xi)
            xi=self.conv_trans[1](xi)
            
            
            xi = xi[:, :, :seq_len]# truncate截断
            xi = xi+x
               
            
            mg=torch.cat((mg, xi),dim=1)
                     
        #mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        mg = mg.permute(0,2,1)
        #print(mg.shape)
        y = self.norm1(mg)
      
        
        return self.norm2(mg + y)  #(batch,seq_len,num_head*feature_size)
        



class MSSD(nn.Module):
    def __init__(self,parameter, conv_kernel=[4,2]):
        super(MSSD, self).__init__()
        self.seq_len = parameter["seq_len"] #输入长度
        self.pred_len = parameter["pred_len"] #预测长度
        
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"]) #embedding
        
        self.linear1 = nn.Linear(int(self.seq_len/3),int(self.pred_len/3))
        self.linear2 = nn.Linear(int(self.seq_len/3),int(self.pred_len/3))
        
        self.sdnet = SDNet(feature_size=parameter["d_model"],conv_kernel=conv_kernel)
        self.proj1 = nn.Linear(parameter["d_model"]*4,parameter["enc_in"])# num_heads =4
        self.proj2 = nn.Linear(int(self.seq_len),int(self.pred_len/3))
        
        self.proj3 = nn.Linear(parameter["enc_in"],parameter["c_out"])
        
    
    def forward(self,x):
        B,seq_len,C=x.shape
        
        #Decomp
        x_a = x[:,0:int(seq_len/3):,] #ascending
        x_p = x[:,int(seq_len/3):int(seq_len/3*2):,] #peak
        x_d = x[:,int(seq_len/3*2):int(seq_len):,]#descending
        
        #Embedding
        x_enc = self.enc_embedding(x_p)
        
    
        #Linear Forecasting
        x_a = self.linear1(x_a.permute(0,2,1)).permute(0,2,1)
        x_d = self.linear2(x_d.permute(0,2,1)).permute(0,2,1)
        
        #print(x_enc.shape)
        #Peak component forecasting
        zeros = torch.zeros((x_enc.shape[0],x_enc.shape[1]*2,x_enc.shape[2]))
        x_enc = torch.cat((x_enc,zeros),dim=1) #padding 0
        x_enc = self.sdnet(x_enc)
        x_enc = self.proj1(x_enc)
        x_enc = self.proj2(x_enc.permute(0,2,1)).permute(0,2,1)
        
        
        #Concat
        out = torch.cat((x_a,x_enc,x_d),dim=1)
        out = self.proj3(out)
        
    
        return out