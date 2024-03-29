# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:37:58 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, series_decomp


class FEDformer(nn.Module):
       
        

    def __init__(self, parameter, version='fourier', mode_select='random', modes=32):
        
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        #self.task_name = configs.task_name
        self.seq_len = parameter["seq_len"]
        #self.label_len = configs.label_len
        self.pred_len = parameter["pred_len"]

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        #Decomp
        kernel_size = parameter["moving_avg"]
        self.decomp = series_decomp(kernel_size)
        
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"])
        self.dec_embedding = DataEmbedding(parameter["dec_in"], parameter["d_model"])
        
        #Fourier
        encoder_self_att = FourierBlock(in_channels=parameter["d_model"],
                                            out_channels=parameter["d_model"],
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        decoder_self_att = FourierBlock(in_channels=parameter["d_model"],
                                            out_channels=parameter["d_model"],
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=parameter["d_model"],
                                                      out_channels=parameter["d_model"],
                                                      seq_len_q=self.seq_len ,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=parameter["n_heads"])
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att,parameter["d_model"], parameter["n_heads"]),
                    parameter["d_model"]
                ) for l in range(parameter["e_layers"])
            ]
        )
        
        self.decoder = Decoder(
                [
                    DecoderLayer(
                            AutoCorrelationLayer(decoder_self_att,parameter["d_model"], parameter["n_heads"]),
                            AutoCorrelationLayer(decoder_cross_att,parameter["d_model"], parameter["n_heads"]),
                        parameter["d_model"],
                        parameter["c_out"]
                    )for l in range(parameter["d_layers"])
                ]
            )
        
        self.proj1 = nn.Linear(parameter["enc_in"],parameter["c_out"])
        self.proj2 = nn.Linear(parameter["seq_len"],parameter["pred_len"])
        self.proj3 = nn.Linear(parameter["d_model"],parameter["c_out"])
        
        
    def forward(self, x_enc):
        
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        #Embedding
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(seasonal_init)
        #enc_out = torch.unsqueeze(enc_out, dim=-1).permute(0,1,3,2)
        #print(enc_out.shape)
        
        
        # enc 
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
    
        #enc_out --> (B,S,D)
        # dec
        seasonal_part, trend_part = self.decoder(enc_out, enc_out, x_mask=None, cross_mask=None,trend=trend_init)
        seasonal_part = self.proj3(seasonal_part)
        dec_out = trend_part + seasonal_part
        dec_out = self.proj1(dec_out)
        dec_out = self.proj2(dec_out.permute(0,2,1))
        dec_out = dec_out.permute(0,2,1)
        
        
        return dec_out # [B, L, D]
        