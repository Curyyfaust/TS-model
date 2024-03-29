
"""
Created on Sun Mar  3 16:11:46 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        #self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        #out = self.ln(out)
        return out
    
class TiDE(nn.Module):  
    
    def __init__(self, parameter, bias=True, feature_encode_dim=2): 
        super(TiDE, self).__init__()
        
        self.seq_len = parameter["seq_len"]  
        self.label_len = parameter["label_len"]
        self.pred_len = parameter["pred_len"]  
        self.hidden_dim= parameter["d_model"]
        self.res_hidden=parameter["d_model"]
        self.encoder_num=parameter["e_layers"]
        self.decoder_num=parameter["d_layers"]
        #self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = parameter["c_out"]
        self.temporalDecoderHidden=4*parameter["d_model"]
        dropout=0.1

        """
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        """
        
        self.feature_dim=4


        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.hidden_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        
       
        
        self.encoders = nn.Sequential(ResBlock(self.hidden_dim+self.feature_encode_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))
        
        self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
        self.temporalDecoder = ResBlock(self.hidden_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        
        
    def forward(self, x_enc):
        # Normalization
        
        # feature = self.feature_encoder(batch_y_mark)
        feature = self.feature_encoder(x_enc)
        #print(feature.shape)
        hidden = self.encoders(torch.cat([x_enc, feature], dim=-1))
        #print(hidden.shape)
        decoded = self.decoders(hidden).reshape(feature.shape[0],-1,self.hidden_dim)[:, -self.seq_len:, :] 
        dec_out = self.temporalDecoder(torch.cat([feature, decoded], dim=-1))
        
        dec_out = self.residual_proj(dec_out.permute(0,2,1)).permute(0,2,1)
        
        return dec_out 
