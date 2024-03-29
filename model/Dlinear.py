
"""
Created on Sat Mar  2 15:13:44 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class Dlinear(nn.Module):

    def __init__(self, parameter):
        
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Dlinear, self).__init__()
        
        self.seq_len = parameter["seq_len"] #输入长度
        self.decompsition = series_decomp(parameter["moving_avg"])
        #self.individual = individual
        self.pred_len = parameter["pred_len"]
        self.channels = parameter["d_model"]
        self.out_channels = parameter["c_out"]
        
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        self.proj1 = nn.Linear(parameter["d_model"],parameter["c_out"])
        
        
    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
       
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0,2,1)
        x = self.proj1(x)
        return x
    
    def forward(self, x_enc):
        
        return self.encoder(x_enc)