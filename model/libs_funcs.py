# -*- coding: utf-8 -*-

import dgl
import dgl.function as fn
from dgl.nn.pytorch import *
from dgl import utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing

import pickle
import networkx as nx

scaler = preprocessing.MinMaxScaler()

def write_data(data, name):
    with open(name + '.bin', 'wb') as f:
        pickle.dump(data, f)
        
def load_data(name):
    try:
        with open(name + '.bin', 'rb') as f:
            data = pickle.load(f)
    except:
        with open(name + '.pickle', 'rb') as f:
            data = pickle.load(f)
    return data

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    return message

def scaling(x):
    if len(x.shape) != 2:
        x = x.reshape(-1, 1)
    return scaler.fit_transform(x)

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj
    
    
