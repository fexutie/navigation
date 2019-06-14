# this is to train the internal weight against decoding, while the improving the behaviour at the same hand, write in a way for full propagation,  another way is to only change weight according to one step  

#  try to make a parallel and continous  training this time,  as in the integrator case, make the life same, so train in batch, use placefielf as position output , try
# to see if we can bridge the integraotor and otehr cases. if the integrator is low rank and not significantly better, so could expect for the cases in middle

# step generate random behaviour and record the action input as forced , then train the hidden state in batches 

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
from torch.utils.data import DataLoader

import POMDPgame_r
from POMDPgame_r import*
import RNN 
from RNN import * 

class PretrainGame(Game): 
    
    def __init__(self, e = 0.2, holes = 3, grid_size = 8, max_size = 20, random_seed = 0, set_reward = 0, time_limit = 100, input_type = 1, batchsize = 1, k_action = 1, k_internal = 1):
        Game.__init__(self, discount = 0.99, grid_size = grid_size, holes = holes, time_limit = time_limit, random_seed = random_seed,
                     set_reward = set_reward, input_type = input_type)
        # need to have randomness
        self.e = e
        self.alpha = 0.5
        self.lam = 0.5 
        self.grid_size = grid_size
        self.net = RNN(9, 512, 4, k_action = k_action, k_internal = k_internal).cuda()
        self.hidden = self.net.initHidden(batchsize = batchsize).cuda()
        self.action = self.net.initAction(batchsize = batchsize).cuda()
        self.Loss = 0
        self.lr = 0
        self.life = 0
        self.succeed = 0
        self.trace = []
        self.hiddens = []
        self.Hiddens = []
        self.y = 0
        self.x = 0
        self.Qs = []
        self.Pos = []
        self.time_limit = time_limit
        self.norm = self.net.h2h.norm()
  
        self.max_size = max_size
       # self.conv1 = nn.Conv2d(1,1,3)
    # take the vectot as max size 
    # absolute value order     
    def placefield(self): 
        field =np.zeros((2, self.max_size + 4))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                field[k, i] =  (i- self.agent.pos[k]) ** 2 
        # gaussian density, but before exponential to help learning identity mapping input to output
        field = - 0.1 * torch.from_numpy(field).resize(1, 2 * (self.max_size + 4)).float()
        return field 
    # target dirac
    def placefield_target(self):
        pos = (self.y, self.x)
        field =np.zeros((2, self.max_size + 4))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                field[k, i] =  (i- pos[k]) ** 2
        # gaussian density, but before exponential to help learning identity mapping input to output
        field = torch.from_numpy(field).resize(1, 2 * (self.max_size + 4)).float()
        return field 
    # topological order
    def placefield_reward(self, pos): 
#         print (pos_, self.set_reward)
#         print (self.grid_size)
        field =np.zeros((2, 19))
#         print (len(field))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                pos_relative = pos[k]
                field[k, i] =  (i- pos_relative) ** 2 
#                 print (i - pos[k])
        # gaussian density, but before exponential to help learning identity mapping input to output
        field = - 0.1 * torch.from_numpy(field).resize(1, 2 * 19).float()
        return field
        
    
    def randomplay(self, state, Action, Action_):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        # store hidden at the instant for afterwards training
        hidden0 = self.hidden.detach()
        # action0 as input for network 
        action0 = Action_
        self.values, self.hidden = self.net(state, hidden0, Action_)
#         self.action = Variable(torch.eye(4)[Action]).resize(1, 4).cuda()  
        return Action, action0

    # in step, the action will be integrated into x and y
    def step(self):  
        """Update state (grid and agent) based on an action"""

        action = self.sample()
        # up

    # here only the network is only trained to predict intial position

    
    def experiment(self, low = 10 , high = 100, e = 0.1, batchsize = 4, size_range = []):   
        # register for datalaoder
        self.e = e
        Actions = []
        Inputs = []
        Targets = []
        steps = np.random.randint(low = low, high = high, size = 1)
        for i in range(batchsize):
            self.reset(size_range = size_range, prob = np.ones(len(size_range))/len(size_range))
            self.y, self.x = self.agent.pos
            actions = []
            inputs = []
            targets = []
            # initialize the random hidden state
            while self.t <steps.item():
                if self.t == 0:
                    action = self.sample()
                # move according to probability
                elif self.t>0:
                    if np.random.random()>0.8:
                        action = self.sample()
                    else:
                        action = action
                # save old position
                y0, x0 = self.y, self.x
                # move
                if action == 0:
                    self.y -= 1
                    # right
                elif action == 1:
                    self.x += 1
                    # down
                elif action == 2:
                    self.y += 1
                    # left
                elif action == 3:
                    self.x -= 1
                self.t += 1
                #prepare network input for next step, register in self.action, size should be 1d  

                # recording sessin, here network input and target is stored, network input is vision and this step action,
                visible_state = self.grid.visible((self.y, self.x)).flatten()
                if self.grid.grid[(self.y, self.x)] < 0:
                    pos_possible = [(a, pos) for (a,pos) in enumerate([(y0-1,x0),(y0,x0+1),(y0+1,x0),(y0,x0-1)]) if self.grid.grid[pos] >= 0]
                    action, pos = pos_possible[np.random.randint(len(pos_possible))]
                    self.y, self.x = pos
                # all flatten
                self.input = torch.FloatTensor(visible_state).resize(9).cuda()
                self.action_ = torch.eye(4)[action].resize(4).cuda()
                actions.append(self.action_)
                inputs.append(self.input) 
                targets.append(self.placefield_target().float().cuda())
                # ensure not inside wall

            Actions.append(torch.stack(actions))
            Inputs.append(torch.stack(inputs))
            Targets.append(torch.stack(targets))
        return Actions, Inputs, Targets
            
    def train(self, lr_rate, Actions, Inputs, Targets, decay = 1e-6, beta = 1e-2):
        # training
        optimizer = torch.optim.Adam(
                [ 
                {'params': self.net.h2p, 'lr': 100 * lr_rate, 'weight_decay':0},
                {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.r2h, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay':0},    
                {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay':decay},
                {'params': self.net.bh, 'lr': lr_rate},
                {'params': self.net.bp, 'lr': 100 * lr_rate},
                ])
        batchsize = len(Inputs)
        # put batch size before sequence length   
        Actions = torch.transpose(torch.stack(Actions), 0, 1).cuda()
        Inputs = torch.transpose(torch.stack(Inputs), 0, 1).cuda()
        Targets = torch.transpose(torch.stack(Targets), 0, 1).cuda()
        hidden0 = self.net.initHidden(batchsize).cuda()
        for epochs in range (50):
                reward_input = torch.stack([self.placefield_reward((9, 6)) for i in range(batchsize)]).squeeze().cuda()
                predicts, hiddens = self.net.forward_sequence(Inputs, hidden0, Actions, reward_input, control = self.reward_control)
                # cross entropy 
                hiddens = torch.stack(hiddens)
                loss_predict = sum([self.net.crossentropy(predict, target, batchsize, beta = beta) for predict,target in zip(predicts, Targets)]) 
                loss_predict.backward(retain_graph = True)
                # gradient clip , from -max to max
                for p, name in zip([self.net.i2h, self.net.a2h, self.net.h2h, self.net.bh],\
                                   ['i2h', 'r2h', 'a2h', 'h2h', 'bh', 'h2p', 'bp'] ):
                        p.grad.data.clamp_(-1, 1)
                optimizer.step()
                # count for print 
                self.Loss += loss_predict
                # clear gradient 
                self.net.zero_grad()
                
    def fulltrain(self, lr_rate = 1e-4, decay = 1e-6, trials = 100, low = 10, high = 100, batchsize = 4, size_range = np.arange(10, 20, 1), beta = 1e-2):
        self.Loss = 0
        # start to experiment
        for i in range(trials):
            Actions, Inputs, Targets = self.experiment(low = low, high = high, batchsize = batchsize, size_range = size_range)
            self.train(lr_rate, Actions, Inputs, Targets, decay = decay, beta = beta)
            if i%50 == 0 and i>0:
                print ('loss for epoch:', self.Loss)
                self.Loss = 0       


