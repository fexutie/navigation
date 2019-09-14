#  This file is to generate the network and train it 
#  Vanilla RNN with the inertia controls internal timescale   
#  random initialization
#  weights used xavier initialization
#  action feedback is by default zero 

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

import os
import psutil
import gc


class RNN(nn.Module):
    def __init__(self, input_size = 9, hidden_size = 512, output_size = 4, predict_size = 5, inertia = 0.5, k_action = 0, reward_size = 38):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Parameter(torch.randn(input_size, hidden_size) * 10 * np.sqrt(2.0/(input_size + hidden_size)))
        self.h2h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1 * np.sqrt(2.0/hidden_size))
        self.h2o = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01 * np.sqrt(2.0/(hidden_size + output_size)))
        self.h2p_rls = nn.Parameter(torch.randn(hidden_size, 2) * 0.01 * np.sqrt(2.0/(hidden_size + 2)))
        self.r2h = nn.Parameter(torch.randn(38, hidden_size) * 0.1 * np.sqrt(2.0/(hidden_size + 38)))
        self.a2h = nn.Parameter(torch.randn(4, hidden_size) * 1 * np.sqrt(2.0/(hidden_size + 4)))
        self.bp_rls = nn.Parameter(torch.zeros(1, 2))
        self.bh = nn.Parameter(torch.zeros(1, hidden_size))
        self.bo = nn.Parameter(torch.zeros(1, output_size))
        # decoder 
        self.I2p = nn.Parameter(torch.randn(hidden_size + input_size + output_size + reward_size, predict_size) * 0.01 * np.sqrt(2.0/(output_size + hidden_size)))
        self.h2a = nn.Parameter(torch.randn(2 * hidden_size + input_size + reward_size, output_size) * 0.01 * np.sqrt(2.0/(output_size + hidden_size)))
        self.bp = nn.Parameter(torch.randn(1, predict_size) * 0 * np.sqrt(2.0/(predict_size + hidden_size)))  
        self.ba = nn.Parameter(torch.randn(1, output_size) * 0 * np.sqrt(2.0/(output_size + hidden_size)))  
        self.r = nn.Parameter(inertia * torch.ones(1, hidden_size))
        self.k_action = k_action
   

    def forward(self, input, hidden, action, reward):
        # dim should be same except catting dimension
        hidden_ = torch.tanh(input.matmul(self.i2h) + hidden.matmul(self.h2h) + self.k_action * action.matmul(self.a2h) + reward.matmul(self.r2h) + self.bh)
        hidden = torch.mul((1 - self.r), hidden_) + torch.mul(self.r, hidden) 
        output = hidden.matmul(self.h2o)+ self.bo
        return output, hidden
    
    def forward_sequence(self, inputs, hidden0, actions, reward, control = 0):
        # dim should be same except catting dimension
        hidden = hidden0
        predicts = []
        hiddens = []
        for input_, action in zip(inputs, actions):  
            hidden_ = F.tanh(input_.matmul(self.i2h) + hidden.matmul(self.h2h) + self.k_action * action.matmul(self.a2h) + reward.matmul(self.r2h) + self.bh)
            hidden = torch.mul((1 - self.r), hidden_) + torch.mul(self.r, hidden) 
            hiddens.append(hidden)
        return hiddens

    def forward_decode(self, inputs, hiddens, actions, reward, control = 0):
        Stim = []
        for input_, action, hidden in zip(inputs, actions, hiddens):
            Input = torch.cat([input_, action, hidden, reward], dim = 1)
            stim_next = Input.matmul(self.I2p) + self.bp 
            Stim.append(stim_next)
        return Stim
    
    def inverse_dynamics(self, inputs, hiddens1, hiddens2, reward, control = 0):
        Acts = []
        for input_, hidden1, hidden2 in zip(inputs, hiddens1, hiddens2):
#             print (input_.size(), action.size(), hidden.size(), reward.size())
            Input = torch.cat([input_, hidden1, hidden2, reward], dim = 1)
            act_next = Input.matmul(self.h2a) + self.ba 
            Acts.append(act_next)
        return Acts
    
    def initHidden(self, batchsize = 1):
        return Variable(torch.randn(batchsize, self.hidden_size))
    
    def initAction(self, batchsize = 1):
        return Variable(torch.zeros(batchsize, self.output_size))
    
    def velocity(self, input, hidden, action, reward):
        hidden_ = torch.tanh(input.matmul(self.i2h) + hidden.matmul(self.h2h) + self.k_action * action.matmul(self.a2h) + reward.matmul(self.r2h) + self.bh)
        return torch.norm((hidden - hidden_).view(-1))
    
    @ staticmethod
    def crossentropy(predict, target, batch_size):
        return torch.mean(- F.softmax(target.view(batch_size,-1), dim = 1) \
                        * torch.log(F.softmax(predict.view(batch_size,-1), dim = 1) + 1e-2)) 
    
# recursive least square is used, mtraix P which is inverse correlaiton of input and beta is updated in a recursive way, pay attention to memory related to matrix inversion  
class LinearRegression(torch.nn.Module):
    def __init__(self, alpha):
        super(LinearRegression, self).__init__()
        self.alpha = alpha
    # choose an alpha to regulizer, alpha determines strength of regulazation  
    # leak not in inversion here 
    def LeastSquare(self, input_, target): 
        self.cov = input_.t().matmul(input_).data.numpy()
        P = torch.inverse(torch.from_numpy(self.cov).float() + self.alpha * torch.eye(input_.size()[1]))
        self.beta = P.matmul(input_.t()).matmul(target)
     
    
class RLS(LinearRegression):
    # get the first N data points and start, data x here is concat version of x and 1
    def __init__(self, alpha, lam = 0.5): 
        LinearRegression.__init__(self, alpha)
        # forget factor 
        self.lam = lam
    # update inverse correlation matrix by new data    
        
    def update_beta(self, x1, y1, trial):
        # kalman gain
        self.cov  = self.lam * self.cov + x1.t().matmul(x1).data.numpy() 
        P = np.linalg.inv(self.cov + self.alpha * np.eye(x1.size()[1]))
        Pr = P @ x1.data.numpy().T
        #  get error
        err = y1.data.numpy() - x1.data.numpy() @ self.beta.data.numpy()
        dbeta = Pr @ err
        self.beta += torch.from_numpy(dbeta).float()
#         err2 = y1.data.numpy() - x1.data.numpy() @ self.beta.data.numpy()
#         print ('triall', trial, 'err', np.sum(err), 'err2', np.sum(err2))
       


