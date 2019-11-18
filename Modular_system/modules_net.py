#  try to write a modular system , modular1, modular2 both receive same input from vision and action, then concatenate the output together
# one thing to be done is to remove the base lne,  check tomorrow the activation of each row, how they compare with each other,  and see if can use balance substraction,  if not trying to use attention gating mechansim

import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init
from torch.nn import DataParallel

import RNN
from RNN import*

import os
import psutil
import gc

import collections
from collections import OrderedDict


class Modules(RNN, nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Modules, self).__init__(input_size, hidden_size, output_size)
        self.rnn1 = RNN(input_size, hidden_size, output_size)
        self.rnn2 = RNN(input_size, hidden_size, output_size)
        self.h2o1 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01 * np.sqrt(2.0/(hidden_size + output_size)))
        self.bo1 = nn.Parameter(torch.zeros(1, output_size))
        self.h2o2 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01 * np.sqrt(2.0 / (hidden_size + output_size)))
        self.bo2 = nn.Parameter(torch.zeros(1, output_size))
        self.has = nn.Parameter(torch.randn(2 * hidden_size, 2) * 0.01 * np.sqrt(2.0 / (hidden_size + output_size)))
        self.bas = nn.Parameter(torch.zeros(1, 2))


    def forward(self, input_, hidden1, hidden2, action, context):
        # dim should be same except catting dimension
        hidden1 = self.rnn1(input_, hidden1, action, context)
        hidden2 = self.rnn2(input_, hidden2, action, context)
        hidden = torch.cat([hidden1, hidden2], dim = 1)
        output1 = hidden1.matmul(self.h2o1) + self.bo1
        output2 = hidden2.matmul(self.h2o2)+ self.bo2
        # module choose probability
        Q_selector = hidden.matmul(self.has) + self.bas
        return output1, output2, hidden1, hidden2, Q_selector
    
    def loadweight(self, weight1, weight2):
#       need to take the state dict as a new dict for updating 
        net_dict1 = torch.load(weight1)
        list_modules = [('h2h', net_dict1['h2h']), ('a2h', net_dict1['a2h']), ('i2h', net_dict1['i2h']), ('r2h', net_dict1['r2h']), ('bh', net_dict1['bh'])]
        select_dict = OrderedDict(list_modules)
        net1 = self.rnn1.state_dict()
        net1.update(select_dict)
        self.rnn1.load_state_dict(net1)
        self.rnn1 = self.rnn1.cpu()
        #load net2
        net_dict2 = torch.load(weight2)
        list_modules = [('h2h', net_dict2['h2h']), ('a2h', net_dict2['a2h']), ('i2h', net_dict2['i2h']), ('r2h', net_dict2['r2h']), ('bh', net_dict2['bh'])]
        select_dict = OrderedDict(list_modules)
        net2 = self.rnn1.state_dict()
        net2.update(select_dict)
        self.rnn2.load_state_dict(net2)
        self.rnn2 = self.rnn2.cpu()


    def initHidden(self, batchsize = 1):
        return Variable(torch.randn(batchsize, self.hidden_size))
    
    def initAction(self, batchsize = 1):
        return Variable(torch.zeros(batchsize, self.output_size))
    
       


