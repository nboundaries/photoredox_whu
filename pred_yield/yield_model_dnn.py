# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 16:54
# @Author  : TXH
# @File    : yield_model_dnn.py
# @Software: PyCharm

import torch
from torch import nn
from gen_dataset import *

class MLP_YIELD(nn.Module):
    '''
    the input feat to this model should be batch_size*D, where D is the result dimension after specified combinations
    '''
    def __init__(self, in_dim=96,
                 fc_dims=[64, 16],
                 act_funs=[nn.ReLU, nn.ReLU],
                 drop_rate=0.2):
        super(MLP_YIELD, self).__init__()
        self.act_funs = act_funs
        self.FC1 = nn.Sequential(
                    nn.Linear(in_dim, fc_dims[0]),
                    act_funs[0](),
                    nn.Dropout(drop_rate)
                    )
        self.FC2 = nn.Sequential(
                    nn.Linear(fc_dims[0], fc_dims[1]),
                    act_funs[1](),
                    nn.Dropout(drop_rate)
                    )
        self.last_layer = nn.Linear(fc_dims[1], 1)

    def _init_parameters(self):
        for p in self.parameters():
            if isinstance(p, nn.Linear):
                # nn.init.xavier_normal(p.weight.data, nn.init.calculate_gain('relu'))
                p.weight.data.normal_(0, 0.1)
                p.bias.data.zero_()

    def forward(self, x):
        h1 = self.FC1(x)
        h2 = self.FC2(h1)
        ys = torch.sigmoid(self.last_layer(h2))
        return ys.float()


class MLP_YIELD2(nn.Module):
    '''
    the input feat to this model should be batch_size*4*D, where D is the original dimension
    '''
    def __init__(self, in_dim=96,
                 fc_dims=[64, 16],
                 act_funs=[nn.ReLU, nn.ReLU],
                 drop_rate=0.2):
        super(MLP_YIELD2, self).__init__()

        self.FC1 = nn.Sequential(
                    nn.Linear(in_dim*2, in_dim),
                    act_funs[0](),
                    nn.Dropout(drop_rate)
                    )
        self.cat_reduce = nn.Sequential(
                    nn.Linear(in_dim, in_dim//2),
                    act_funs[0](),
                    nn.Dropout(drop_rate)
                    )
        self.reg_reduce = nn.Sequential(
                    nn.Linear(in_dim, in_dim//2),
                    act_funs[0](),
                    nn.Dropout(drop_rate)
                    )
        self.FC2 = nn.Sequential(
                    nn.Linear(2*in_dim, in_dim//4),
                    act_funs[1](),
                    nn.Dropout(drop_rate)
                    )
        self.last_layer = nn.Linear(in_dim//4, 1)

    def forward(self, x):
        h1 = self.FC1(torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1))
        cat = self.cat_reduce(x[:, 2, :])
        reg = self.reg_reduce(x[:, 3, :])
        h2 = self.FC2(torch.cat([h1, cat, reg], dim=-1))
        ys = torch.sigmoid(self.last_layer(h2))
        return ys.float()


class MLP_YIELD_Classifier(nn.Module):
    def __init__(self, in_dim=96,
                 fc_dims=[64, 16],
                 act_funs=[nn.ReLU, nn.ReLU],
                 drop_rate=0.2):
        super(MLP_YIELD_Classifier, self).__init__()

        self.FC1 = nn.Sequential(
                    nn.Linear(in_dim, fc_dims[0]),
                    act_funs[0](),
                    nn.Dropout(drop_rate)
                    )
        self.FC2 = nn.Sequential(
                    nn.Linear(fc_dims[0], fc_dims[1]),
                    act_funs[1](),
                    nn.Dropout(drop_rate)
                    )
        self.last_layer = nn.Linear(fc_dims[1], 3)

    def forward(self, x):
        h1 = self.FC1(x)
        h2 = self.FC2(h1)
        ys = self.last_layer(h2)
        return ys.float()

def _init_parameters(model):
    for p in model.parameters():
        if isinstance(p, nn.Linear):
            # nn.init.xavier_normal(p.weight.data, nn.init.calculate_gain('relu'))
            p.weight.data.normal_(0, 0.01)
            p.bias.data.zero_()

if __name__ == '__main__':
    # if categorical is False, do regression
    model = MLP_YIELD()
    model._init_parameters()











