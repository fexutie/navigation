# this is to pretrain the network as evolution/development.  It should learn a curiosity module which let it to predict next state stimulus and its action done  

#  for inverse dynanmics part , The current and next state will be put together and using a linear layer to output the action used, it can be also composed of current and next next state  

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
    
    def __init__(self, e = 0.2, holes = 3, grid_size = 8, max_size = 20, random_seed = 0, set_reward = 0, time_limit = 100, input_type = 1, batchsize = 1):
        Game.__init__(self, discount = 0.99, grid_size = grid_size, holes = holes, time_limit = time_limit, random_seed = random_seed,
                     set_reward = set_reward, input_type = input_type)
        # need to have randomness
        self.e = e
        self.alpha = 0.5
        self.lam = 0.5 
        self.grid_size = grid_size
        self.net = RNN(9, 512, 4, k_action = 1).cuda()
#         self.decoder = decoder().cuda()
        self.hidden = self.net.initHidden(batchsize = batchsize).cuda()
        self.action = self.net.initAction(batchsize = batchsize).cuda()
        self.Loss = 0
        self.lr = 0
        self.life = 0
        self.succeed = 0
        self.trace = []
        self.hiddens = []
        self.Hiddens = []
        self.Qs = []
        self.Pos = []
        self.time_limit = time_limit
        self.norm = self.net.h2h.norm()
        self.y_mid = 0
        self.x_mid = 0
        for i in range(len(set_reward)):
            self.y_mid += self.set_reward[i][0] 
            self.x_mid += self.set_reward[i][1] 
        self.y_mid /= len(set_reward)
        self.x_mid /= len(set_reward)
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
    # topological order
    def placefield_reward(self, pos): 
        pos_ = (pos[0] - self.y_mid, pos[1] - self.x_mid)
#         print (pos_, self.set_reward)
#         print (self.grid_size)
        field =np.zeros((2, 19))
#         print (len(field))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                pos_relative = pos[k] * 19./(self.grid_size[1] + 4)
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

    # because of stop gradient, train only one step before 
    def step(self):  
        self.t += 1
        """Update state (grid and agent) based on an action"""
        pos0 = self.agent.pos
        action = self.sample()
        self.agent.act(action)
        pos1 = self.agent.pos
         # wall detection  
        return pos0, action, pos1
    
    def wall_detection(self):
        if self.agent.pos[0] == 2:
            self.Stim = torch.eye(5)[1]
        elif self.agent.pos[0] == 2 + self.grid_size[0] - 1:
            self.Stim = torch.eye(5)[2]
        elif self.agent.pos[1] == 2:
            self.Stim = torch.eye(5)[3]
        elif self.agent.pos[1] == 2 + self.grid_size[1] - 1:
            self.Stim = torch.eye(5)[4]
            
    # here only the network is only trained to predict intial position 

    # random walk inside maze, record action and stimulus 
    def experiment(self, low = 10 , high = 100, e = 0.1, batchsize = 4, size_range = np.arange(5, 56, 10)):   
        # register for datalaoder
        self.e = e
        Actions = []
        Inputs = []
        Targets = []
        steps = np.random.randint(low, high = high, size = 1)
        mid_point = np.random.randint(0, high = steps, size = 1)
        for i in range(batchsize):
            self.reset(size_range = size_range)
            actions = []
            inputs = []
            targets = []
            # initialize the random hidden state 
            self.Stim = torch.eye(5)[0]
            while self.t <steps.item():
                pos0, action, pos1 = self.step()
                #prepare network input for next step, register in self.action, size should be 1d  
                self.action_ = torch.eye(4)[action].resize(4).cuda()
                # recording sessin, here network input and target is stored, network input is vision and this step action, 
                vision = self.visible_state
                # all flatten
                self.input = torch.FloatTensor(self.visible_state).resize(9).cuda()
                actions.append(self.action_)
                inputs.append(self.input) 
                # target as memory of wall, start with 0, means no memory
                self.wall_detection()
                targets.append(self.Stim.float().cuda())
                # ensure not inside wall
                if self.grid.grid[pos1] < 0:
                    self.agent.pos = pos0
                    y,x = pos0      
                    pos_possible = [(a, pos) for (a,pos) in enumerate([(y-1,x),(y,x+1),(y+1,x),(y,x-1)]) if self.grid.grid[pos] >= 0]
                    action, pos = pos_possible[np.random.randint(len(pos_possible))]
                    self.agent.pos = pos
            Actions.append(torch.stack(actions))
            Inputs.append(torch.stack(inputs))
            Targets.append(torch.stack(targets))
        return Actions, Inputs, Targets
    
#     def forward_model(self, Actions, Inputs, ):
#         hiddens = self.net.forward_sequence(Inputs, hidden0, Actions, reward_input, control = self.reward_control)
      
        
#     def inverse_model(self):
#         actions = self.net.inverse_sequence(hiddens)
        
            
    def train(self, lr_rate, Actions, Inputs, Targets, decay = 1e-7, batchsize = 8):
        # training
        optimizer = torch.optim.Adam(
                [ 
#                 {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay':0},
#                 {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay':0},    
                {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay':decay},
                {'params': self.net.bh, 'lr': lr_rate},
                {'params': self.net.h2a, 'lr': lr_rate},
                {'params': self.net.I2p, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.bp, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.ba, 'lr': lr_rate, 'weight_decay':0},
                ])
        # put batch size before sequence length   
        Actions = torch.transpose(torch.stack(Actions), 0, 1).cuda()
        Inputs = torch.transpose(torch.stack(Inputs), 0, 1).cuda()
        Targets = torch.transpose(torch.stack(Targets), 0, 1).cuda()
        hidden0 = self.net.initHidden(batchsize).cuda()
        for epochs in range (50):
                reward_control = np.random.randint(0, 2)
                if reward_control == 0:
                    reward_input = torch.stack([self.placefield_reward(self.Set_reward[0]) for i in range(batchsize)]).squeeze().cuda()
                else:
                    reward_input = torch.stack([self.placefield_reward(self.Set_reward[1]) for i in range(batchsize)]).squeeze().cuda()
                hiddens = self.net.forward_sequence(Inputs, hidden0, Actions, reward_input, control = self.reward_control)
                Predicts = self.net.forward_decode(Inputs, hiddens, Actions, reward_input, control = self.reward_control)
                Actions_ = self.net.inverse_dynamics(Inputs[:-1], hiddens[:-1], hiddens[1:], reward_input, control = self.reward_control)
                # cross entropy 
                hiddens = torch.stack(hiddens)
                loss_predict = sum([self.net.crossentropy(predict, target, batchsize) for predict,target in zip(Predicts, Targets)]) 
                loss_inverse = sum([self.net.crossentropy(act0, act1, batchsize) for act0, act1 in zip(Actions_, Actions[:-1])]) 
                loss = 0.7 * loss_predict + 0.3 * loss_inverse
#                 loss = loss_inverse
                loss.backward(retain_graph = True)
                # gradient clip , from -max to max
                for p, name in zip([self.net.h2h, self.net.bh],\
                                   ['h2h', 'bh'] ):
                        p.grad.data.clamp_(-1, 1)
                optimizer.step()
                # count for print 
                self.Loss += loss_predict
                # clear gradient 
                self.net.zero_grad()
    
    def test(self, Actions, Inputs, Targets, decay = 1e-6, batchsize = 1):
        # training
        Actions = torch.transpose(torch.stack(Actions), 0, 1).cuda()
        Inputs = torch.transpose(torch.stack(Inputs), 0, 1).cuda()
        Targets = torch.transpose(torch.stack(Targets), 0, 1).cuda()
        hidden0 = self.net.initHidden(batchsize).cuda()
        for epochs in range (1):
                reward_control = np.random.randint(0, 2)
                if reward_control == 0:
                    reward_input = torch.stack([self.placefield_reward(self.Set_reward[0]) for i in range(batchsize)]).view(1, -1).cuda()
                else:
                    reward_input = torch.stack([self.placefield_reward(self.Set_reward[1]) for i in range(batchsize)]).view(1, -1).cuda()
                hiddens = torch.stack(self.net.forward_sequence(Inputs, hidden0, Actions, reward_input, control = self.reward_control))
                predicts = self.net.forward_decode(Inputs, hiddens, Actions, reward_input, control = self.reward_control)
                # cross entropy 
        return predicts, Targets, hiddens 
                
    def fulltrain(self, lr_rate = 1e-4, decay = 1e-6, trials = 100, low = 10, high = 100, batchsize = 8, size_range = np.arange(5, 56, 10)):
        self.Loss = 0
        # start to experiment
        for i in range(trials):
            Actions, Inputs, Targets = self.experiment(low = low, high = high, batchsize = batchsize, size_range = size_range)
            self.train(lr_rate, Actions, Inputs, Targets, decay = decay,  batchsize = batchsize)
            if i%50 == 0 and i>0:
                print ('loss for epoch:', self.Loss)
                self.Loss = 0       
                
    def fulltest(self, low = 10, high = 100, batchsize = 1, size_range = [3]):
        self.Loss = 0
        # start to experiment
        for i in range(1):
            Actions, Inputs, Targets = self.experiment(low = low, high = high, batchsize = batchsize, size_range = size_range)
            predicts, targets, hiddens = self.test(Actions, Inputs, Targets, batchsize = batchsize)
        return predicts, targets, hiddens
     

