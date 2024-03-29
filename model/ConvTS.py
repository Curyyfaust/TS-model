
"""
Created on Sat Mar  2 18:20:34 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, input_dim, hid_dim,output_dim, kernel_size):
        super(ConvBlock2d, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.kernel_size=kernel_size  
        
        
        self._build()
     
    def _build(self):
        self.conv1=nn.Conv2d(self.input_dim,self.hid_dim,kernel_size=1)
        self.conv2=nn.Conv2d(self.hid_dim,self.output_dim,kernel_size=1)
        
        
    def forward(self, x):
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0],-1,x.shape[2])
       
        return x

class ConvBlock1d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvBlock1d, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size=kernel_size  
        
        
        self._build()
     
    def _build(self):
        self.conv=nn.Conv1d(self.input_dim,self.output_dim,kernel_size=self.kernel_size)
        self.conv_trans=nn.ConvTranspose1d(self.output_dim,self.input_dim,kernel_size=self.kernel_size)
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.conv_trans(x)
       
        return x
    
class ConvTS(nn.Module):
    def __init__(self,parameter,chunk_size=24):
        super(ConvTS, self).__init__()

        self.input_len = parameter["seq_len"]
        self.output_len = parameter["pred_len"]

        self.chunk_size = chunk_size
        #assert(input_len % chunk_size == 0)
        self.num_chunks = parameter["seq_len"] // chunk_size
        
        self.input_dim = parameter["d_model"]
        self.output_dim = parameter["c_out"]

        self.layer2d_1=ConvBlock2d(input_dim=self.chunk_size,
                               hid_dim=self.chunk_size*int(self.num_chunks/2),
                               output_dim=self.chunk_size*self.num_chunks,
                               kernel_size=1)
        
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        self.layer1d_1=ConvBlock1d(input_dim=self.chunk_size,                            
                               output_dim=self.chunk_size,
                               kernel_size=self.num_chunks)
        
        self.layer2d_2=ConvBlock2d(input_dim=self.chunk_size,
                               hid_dim=self.chunk_size*int(self.num_chunks/2),
                               output_dim=self.chunk_size*self.num_chunks,
                               kernel_size=1)
        
        
        self.layer1d_2=ConvBlock1d(input_dim=self.chunk_size,                            
                               output_dim=self.chunk_size,
                               kernel_size=self.num_chunks)
        
        
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)
        
        self.up=nn.Upsample(scale_factor=2, mode='nearest')
    
        self.ar1 = nn.Linear(self.input_len, self.output_len)
        self.ar2 = nn.Linear((self.input_len+self.chunk_size)*2, self.output_len)
        self.proj1 = nn.Linear(parameter["d_model"],parameter["c_out"])

    def forward(self, x):
        B, T, N = x.size()

        highway = self.ar1(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        # 2d 1
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)  
        x11 = self.layer2d_1(x1)
        x12 = self.layer1d_1(x1)
        x1 = torch.cat([x11,x12],dim=1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1) #
        
        
        # 2d 2
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = torch.cat([self.layer2d_2(x2),self.layer1d_2(x2)],dim=1)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)
        
        
        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1)
        x3 = self.ar2(x3)
        x3 = x3.permute(0, 2, 1)
        
        out = x3 + highway
        
        #print(out.shape)
        out = self.proj1(out)
        
        
        return out

