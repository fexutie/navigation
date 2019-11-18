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

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML

import POMDPgame_bars
from POMDPgame_bars import*

import POMDPgame_basic
from POMDPgame_basic import*

import POMDPgame_holes
from POMDPgame_holes import*


import RNN
from RNN import *

import navigation2
from navigation2 import*

import Tests
from Tests import*

import sklearn
from sklearn.svm import SVC

import scipy
from scipy.spatial import distance
from scipy import signal


# A complete experiment including pretraining , decoding training, and q learning  
class MultipleTasks():
    def __init__(self, weight_write, task =  'basic', noise = 0.0, gpu = 0):
        if task == 'basic':
            self.game = CreateGame(GameBasic, holes= 0, noise = noise, task = task, gpu = gpu)
        if task == 'hole':
            self.game = CreateGame(GameHole, holes= 50, noise = noise, task = task, gpu = gpu)
        if task == 'bar':
            self.game = CreateGame(GameBar, holes= 0, noise = noise, task = task, gpu = gpu)
        if task == 'scale':
            self.game = CreateGame(GameScale, holes=0, noise = noise, task = task, gpu = gpu)
        if task == 'scale_x':
            self.game = CreateGame(GameScale_x, holes=0, noise = noise, task = task, gpu = gpu)
        if task == 'scale_y':
            self.game = CreateGame(GameScale_y, holes=0, noise = noise, task = task, gpu = gpu)
        if task == 'implicit':
            self.game = CreateGame(GameImplicit, holes=0, noise = noise, implicit = True, gpu = gpu)
        self.weight = weight_write
        self.task = task
        # plt.matshow(game.grid.grid)

    def loadweight(self, weight_load):
#       need to take the state dict as a new dict for updating 
        net_dict = torch.load(weight_load)
        list_modules = [('h2h', net_dict['h2h']), ('a2h', net_dict['a2h']), ('i2h', net_dict['i2h']), ('r2h', net_dict['r2h']), ('bh', net_dict['bh'])]
        select_dict = OrderedDict(list_modules)
        net = self.pregame.net.state_dict()
        net.update(select_dict)
        self.pregame.net.load_state_dict(net)
        torch.save(self.pregame.net.state_dict(), self.weight) 
    
        
    def pretrain(self, trial, weight = None, lr = 1e-5, pretrain = True):  
        # start a pretrained game  
        self.pregame.net.cuda()
        if pretrain == True:
            lr = float(lr)
            if weight != None:
                self.pregame.net.load_state_dict(torch.load(weight))
            self.pregame.fulltrain(lr_rate = lr, trials = int(1e3), batchsize = 4)
        print ('pretrain end', torch.norm(self.pregame.net.h2h))
        if pretrain == True:
            torch.save(self.pregame.net.state_dict(), self.weight[:-1]+'{}'.format(trial))
        else:
            torch.save(self.pregame.net.state_dict(), self.weight+'{}'.format(trial))
        if pretrain == True and trial <= 10:
            self.weight = self.weight[:-1]+'{}'.format(trial)
        elif pretrain == True and trial > 10:
            self.weight = self.weight[:-2]+'{}'.format(trial)
        elif pretrain == False:
            self.weight = self.weight +'{}'.format(trial)
            
              
    def decode(self, weight = None, size_range = [15], size_test = [15], epsilon = 0):
        if weight != None:
            self.game.net.load_state_dict(torch.load(weight))
        else:
            self.game.net.load_state_dict(torch.load(self.weight))
        rls_q = RLS(1e2)
        rls_sl = RLS(1e2)
        def precision(size = 15, reward_control = 0):
            dist, decode, visit, Cor_y, Cor_x = decodetest(self.game, reward_control= reward_control, epsilon = 0, size = size)
            prec0 = np.mean(dist)
            return np.mean(decode/visit), decode/visit
        Prec = 0
        # iterations are number of turns for decoder update, epochs are how many turns of games at each update  
        Prec_matrix = np.zeros((15, 15))
        for reward_control in [0, 1]:
            self.game.experiment(rls_q, rls_sl, iterations = 50, epochs = 10, epsilon = epsilon, train_hidden = False, train_q = False, size_range = size_range, test = True, decode = True, reward_control = reward_control)
            prec, prec_matrix =  precision(size = size_test[0], reward_control = reward_control)
            Prec += prec
            Prec_matrix += prec_matrix
            print(self.game.reward_control, prec)
        # tested on size 15
        Prec = Prec/2
        Prec_matrix = Prec_matrix/2
        print ('decode train finish', Prec)
        return Prec, Prec_matrix

    
        
    def qlearn(self, task, weight_read, weight_write, episodes = 10, save = True, size_train =  [15], \
        size_test = [15], test_only = False, noise = 0.0, k_action = 1, epochs = 10, lr_rate = 1e-5, batch_size = 8):
        self.game.net.load_state_dict(torch.load(weight_read))
#         if h2o == True:
#             self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * 0.01 * np.sqrt(2.0/(512 + 4)))
#             self.game.net.h2o = self.game.net.h2o.cuda()
        self.game.net.k_action = k_action    
        e_rate = [noise for r in range(episodes)]
        rls_sl = RLS(1e2)
        # q leanring phase
        for n,e in enumerate(e_rate):
            prob = np.ones(len(size_train)) 
            prob = prob/np.sum(prob)
#             self.game.seed_range = 2 + (n//10) * 10
            self.game.seed_range = 1e5
            if test_only == False:
                self.game.experiment(rls_sl, epochs = 10, epsilon=noise, size_range=size_train,  lr_rate = lr_rate, batch_size = batch_size)
                if save == True:
                    torch.save(self.game.net.state_dict(), weight_write + '_{}'.format(n))
            # plt.matshow(self.game.grid.grid)
            rewards = Test(task, self.game, weight= weight_write + '_{}'.format(n), size = size_test[0], limit_set = 2, test_size = self.game.size//10  - 1)








