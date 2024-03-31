
"""
Created on Fri Mar 22 15:56:10 2024

@author: Yipu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class Projector(nn.Module):
    
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        
        
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y
"""
x=torch.randn(32,96,8)
std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
y=model(x,std_enc)==>(32,1)

"""


class NSTransformer(nn.Module):
   

    def __init__(self, parameter):
        super(NSTransformer, self).__init__()
        #self.task_name = configs.task_name
        self.pred_len = parameter["pred_len"]
        
        #self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"])
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(), parameter["d_model"], parameter["n_heads"]),
                    parameter["d_model"]
                ) for l in range(parameter["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(parameter["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(),parameter["d_model"], parameter["n_heads"]),
                        AttentionLayer(
                            DSAttention(),parameter["d_model"], parameter["n_heads"]),
                        parameter["d_model"]
                    )for l in range(parameter["d_layers"])
                ],
                norm_layer=torch.nn.LayerNorm(parameter["d_model"]),
                projection=nn.Linear(parameter["d_model"], parameter["c_out"], bias=True)
            )
        
        self.proj1 = nn.Linear(parameter["seq_len"],parameter["pred_len"])
        #self.proj2 = nn.Linear(parameter["d_model"],parameter["c_out"])
        
        self.tau_learner = Projector(enc_in=parameter["enc_in"], seq_len=parameter["seq_len"], hidden_dims=parameter["hidden_dims"],
                                     hidden_layers=parameter["hidden_layers"], output_dim=1)
        self.delta_learner = Projector(enc_in=parameter["enc_in"], seq_len=parameter["seq_len"], hidden_dims=parameter["hidden_dims"],
                                     hidden_layers=parameter["hidden_layers"], output_dim=1)
    
    
    
    def forward(self, x_enc):
        
        x_raw = x_enc.clone().detach()
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc# B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()# B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        #dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(enc_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = dec_out * std_enc + mean_enc
        
        dec_out = self.proj1(dec_out.permute(0,2,1)).permute(0,2,1)
        #dec_out = self.proj2(dec_out)
        
        return dec_out
