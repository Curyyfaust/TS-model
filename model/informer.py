# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:08:00 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
#from math import sqrt


class informer(nn.Module):
    """
    with O(Llog(l)) complexity
    """

    def __init__(self, parameter):
        super(informer, self).__init__()
        #self.task_name = configs.task_name
        self.pred_len = parameter["pred_len"]
        
        #self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"])
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(), parameter["d_model"], parameter["n_heads"]),
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
                            ProbAttention(),parameter["d_model"], parameter["n_heads"]),
                        AttentionLayer(
                            ProbAttention(),parameter["d_model"], parameter["n_heads"]),
                        parameter["d_model"]
                    )for l in range(parameter["d_layers"])
                ],
                norm_layer=torch.nn.LayerNorm(parameter["d_model"]),
                projection=nn.Linear(parameter["d_model"], parameter["c_out"], bias=True)
            )
        
        self.proj = nn.Linear(parameter["seq_len"],parameter["pred_len"])
    
    
    def forward(self, x_enc):
        # Embedding
        enc_out=self.enc_embedding(x_enc)
        
        #Encoder
        #enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        #dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(enc_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.proj(dec_out.permute(0,2,1)).permute(0,2,1)
        
        return dec_out # [B, L, D]