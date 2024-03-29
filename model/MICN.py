
"""
Created on Sun Mar  3 10:54:17 2024

@author: Yipu
"""

import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[3], conv_kernel=[2],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1))
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        # merge
        mg = torch.tensor([])
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)
    
"""
model=MIC(feature_size=8)
y=model(x)
y.shape-->(32,96,8)

"""


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[3], c_out=1,
                 conv_kernel=[2], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)
    
class MICN(nn.Module):
    def __init__(self, parameter, conv_kernel=[12, 16]):
        
        super(MICN, self).__init__()

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        
        
        self.seq_len = parameter["seq_len"] #输入长度
        self.pred_len = parameter["pred_len"]#预测长度
        
        for i in conv_kernel:
            decomp_kernel.append(i + 1)
            isometric_kernel.append((parameter["seq_len"] +parameter["pred_len"] + i) // i)
        
        
        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # Forecasting
        self.conv_trans = SeasonalPrediction(embedding_size=parameter["d_model"],
                                            n_heads=parameter["n_heads"],
                                            d_layers=parameter["d_layers"],
                                            c_out=parameter["c_out"])
        self.proj1 = nn.Linear(parameter["seq_len"],parameter["pred_len"])
        self.proj2 = nn.Linear(parameter["d_model"],parameter["c_out"])
        self.proj3 = nn.Linear(parameter["seq_len"],parameter["pred_len"])
        
        
    
    def forward(self,x_enc):
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        #seasonal,trend
        trend = self.proj1(trend.permute(0, 2, 1)).permute(0, 2, 1) #forecasting trend
        trend = self.proj2(trend)
        
        dec_out = self.conv_trans(x_enc)
        dec_out = self.proj3(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out + trend
        
        return dec_out