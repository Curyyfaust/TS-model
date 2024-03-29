
"""
Created on Sun Mar  3 13:19:27 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Convblocks import Inception_Block_V1
from layers.Embed import DataEmbedding

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    for i in range(k):
        if top_list[i]==0:
            top_list[i]=1
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, parameter):
        super(TimesBlock, self).__init__()
        self.seq_len = parameter["seq_len"]
        self.pred_len = parameter["pred_len"]
        self.k = 2
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(parameter["d_model"], parameter["d_model"]*4),
            nn.GELU(),
            Inception_Block_V1(parameter["d_model"]*4, parameter["d_model"])
        )

    def forward(self, x):
        B, T, N = x.size()
        #print(T)
        period_list, period_weight = FFT_for_Period(x, self.k)
        period_list = [2,2]
        #print(period_list)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            
            
            
            out = out.reshape(B, (length) // period, period,N).permute(0,3,2,1)
            # 2D conv: from 1d Variation to 2d Variation
            #print(out.shape)
            out = self.conv(out)
            # reshape back
            
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    

    def __init__(self, parameter):
        super(TimesNet, self).__init__()
        
        self.seq_len = parameter["seq_len"]
        self.label_len = parameter["label_len"]
        self.pred_len = parameter["pred_len"]
        self.model = nn.ModuleList([TimesBlock(parameter)
                                    for _ in range(parameter["e_layers"])])
       
        self.layer = parameter["e_layers"]
        self.layer_norm = nn.LayerNorm(parameter["d_model"])
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(parameter["d_model"], parameter["c_out"], bias=True)
        self.projection2 = nn.Linear(parameter["seq_len"]+parameter["pred_len"],parameter["pred_len"])
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"])
       
        

    def forward(self, x_enc):
        #Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        dec_out = self.projection2(dec_out.permute(0,2,1)).permute(0,2,1)

        
        return dec_out
