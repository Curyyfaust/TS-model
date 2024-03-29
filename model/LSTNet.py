
"""
Created on Sun Mar  3 22:17:18 2024

@author: Yipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTNet(nn.Module):
    def __init__(self, parameter):
        super(LSTNet, self).__init__()
        #self.use_cuda = args.cuda
        self.seq_len = parameter["seq_len"] 
        self.pred_len = parameter["pred_len"]
        
        #self.m = data.m
        self.hidR = 128
        self.hidC = 64
        self.hidS = 128
        self.kernel = 1
        self.skip = 2
        self.hw = 2
        self.pt = int((self.seq_len - self.kernel)/self.skip)
        
        """
        self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        """
        
        self.conv1 = nn.Conv2d(parameter["d_model"], self.hidC, kernel_size = self.kernel)
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(0.1)
        
        self.GRUskip = nn.GRU(self.hidC, self.hidS)
        self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, parameter["pred_len"])
        
        self.highway = nn.Linear(self.hw, parameter["pred_len"])
        self.linear2 = nn.Linear(1,parameter["c_out"])
        self.linear3 = nn.Linear(parameter["d_model"],parameter["c_out"])
        
       
    
    
    def forward(self, x):
        
        batch_size = x.size(0)
        
        #CNN
        c = x.view(batch_size, x.size(-1), self.seq_len, 1)
        c = self.conv1(c)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))
    

        
        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            #print(s.shape)
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        res = res.view(res.shape[0],res.shape[1],1)
        res = self.linear2(res)
        #print(res.shape)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1)
            z = self.highway(z)
            z = self.linear3(z.permute(0,2,1))
            res = res+z
            
        
        return res