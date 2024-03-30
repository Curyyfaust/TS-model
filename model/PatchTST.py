
"""
Created on Sun Mar 24 14:07:50 2024

@author: Yipu
"""

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding,DataEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        nf=n_vars*d_model
        """
        
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        #print(x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class PatchTST(nn.Module):
    

    def __init__(self, parameter, patch_len=8, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        #self.task_name = configs.task_name
        self.seq_len= parameter["seq_len"]
        self.pred_len = parameter["pred_len"]
        padding = stride

        # patching and embedding
        self.enc_embedding = DataEmbedding(parameter["enc_in"], parameter["d_model"])
        self.patch_embedding = PatchEmbedding(
            parameter["d_model"], patch_len, stride, padding, 0.1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(), parameter["d_model"], parameter["n_heads"]),
                    parameter["d_model"]
                ) for l in range(parameter["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(parameter["d_model"])
        )
        
        self.head_nf = parameter["d_model"] * \
                       int((parameter["seq_len"] - patch_len) / stride + 2)
        
        #Decoder
        self.head = FlattenHead(parameter["seq_len"],self.head_nf, parameter["pred_len"])
        self.proj1 = nn.Linear(parameter["d_model"],parameter["c_out"])
        
        
    def forward(self, x_enc):
        # do patching and embedding
        #print(self.head_nf)
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        
        enc_out, n_vars = self.patch_embedding(enc_out)
        
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.proj1(dec_out)

        return dec_out
        