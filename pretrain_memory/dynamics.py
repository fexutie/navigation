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

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
from IPython.display import HTML

import pretrain
from pretrain import *

import pandas as pd

import navigation2
from navigation2 import *

import sklearn
from sklearn.svm import SVC

import scipy
from scipy.spatial import distance
from scipy import signal


# attention to noise level, here corresponed to pretraining , so set noise to 1 
def trajectory(game, pos0, reward_control = 0, init_hidden = True, hidden = torch.zeros(512, 512), size = 19, test = 2):
    game.reset(set_agent = pos0, reward_control = reward_control, size = size, limit_set = 32, test = test)
    done = False
    if init_hidden == True:
        game.hidden = game.net.initHidden()
    else:
        game.hidden = hidden.clone()
    hidden0 = game.hidden.clone()
    Hidden = []
    dH = []
    Action = []
    State = []
    Pos = []
    Pos.append(game.agent.pos)
    while not done:
        pos0, state, reward, done = game.step(game.maxplay, epsilon = 0.00, test=True) # Down
        Hidden.append(game.hidden.data.numpy().squeeze())
        dH.append(torch.norm(game.hidden - hidden0))
        Pos.append(game.agent.pos)
        Action.append(np.argmax(game.action.data.numpy()))
        State.append(state)
        hidden0 = game.hidden.clone()
    return Pos, np.array(Hidden), np.array(dH), np.array(Action), reward

def trajectory_empty(pos0, game, Stim, reward_control = 0, action_ = 0, e = 0, open_loop = True):
    game.reset(reward_control = reward_control, size = 15)
    done = False
    game.hidden = game.net.initHidden()
    game.action = torch.zeros(1, 4)
    Hidden = []
    Action = []
    Pos = []
    dH = []
    hidden0 = game.hidden.clone()
    game.agent.pos = pos0 
    stim = game.visible_state 
#     print (game.visible_state)
    y, x = 0, 0
    action = action_
    for stim in Stim:
        if open_loop == True:
            action = game.step_empty(stim = stim, action_ = action_, epsilon = e, open_loop = open_loop) # Down
        else:
            action = game.step_empty(stim = stim, action_ = action, epsilon = e, open_loop = open_loop) # Down
        # up 
        if action == 0: y -= 1
        # right
        elif action == 1: x += 1
        # down
        elif action == 2: y += 1
        # left
        elif action == 3: x -= 1
        game.agent.pos = (y, x)
        Pos.append(game.agent.pos)
        Hidden.append(game.hidden.clone().data.numpy().squeeze()) # need copy , avoid same adress 
        dH.append(torch.norm(game.hidden - hidden0))
        Action.append(action)
        hidden0 = game.hidden.clone()
    dH = [dh.data.numpy() for dh in dH]
    return Pos, np.array(Hidden), np.array(dH), np.array(Action)


def trajectory_room(pos0, game, T_total = 50, reward_control = 0, epsilon = 0, start = []):
    game.reset(set_agent = (start[0], start[1]), reward_control = reward_control, size = game.size)
    done = False
    game.hidden = game.net.initHidden()
    game.action = torch.zeros(1, 4)
    Hidden = []
    Action = []
    Pos = []
    dH = []
    hidden0 = game.hidden.clone()
    game.agent.pos = pos0 
    game.grid.grid[2:game.size + 2, 2:game.size + 2] = 0
    for i in range(T_total):
        pos0, state, reward, done = game.step(game.maxplay, epsilon = epsilon, test=True)
        Pos.append(game.agent.pos)
        Hidden.append(game.hidden.clone().data.numpy().squeeze()) # need copy , avoid same adress 
        dH.append(torch.norm(game.hidden - hidden0))
        Action.append(np.argmax(game.action.data.numpy()))
        hidden0 = game.hidden.clone()
    dH = [dh.data.numpy() for dh in dH]
    return Pos, np.array(Hidden), np.array(dH), np.array(Action)

class PCA():
    def __init__(self, size = 15, reward_control = 0, trial = 0, file = 1, iters1 = 0, iters2 = 9, epsilon = 1):
        self.epsilon = epsilon
        self.Hiddens = np.zeros((225, 50, 512))
        self.reward_control = reward_control
        self.weight = 'weights2/rnn_1515tanh512_checkpoint{}'.format(trial)
        self.game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 4 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = 0, action_control = 1, 
                                 discount = 0.9, alpha = 1, time_limit=100, lam = 0.5)
        self.game.net.load_state_dict(torch.load(self.weight))
        self.trial = trial
    # record hidden activity, attention take test = true
    def record(self, T_duration = 50):
        for j in range (2,17):
            for i in range (2,17):
                self.game.reset(set_agent = (j,i), reward_control = self.reward_control, size = 15)
                done = False
                self.game.hidden = self.game.net.initHidden()
                for t in range(T_duration):
                    pos0, state, reward, done = self.game.step(self.game.maxplay, epsilon = self.epsilon, test = True) # Down
                    self.Hiddens[(j-2) * 15 + (i-2), t] = (self.game.hidden.data.numpy().squeeze())
                    if done == True:
                        self.game.reset(reward_control = self.reward_control, size = 15)
                        done = False 
    # put every row(time) together into one long trajectory, which means concatenation along rows - number of neurons
    def pca(self, T_duration = 50):
        self.record(T_duration = T_duration)
        Trajectory = self.Hiddens.reshape(225 * 50, 512)
        #  take correlation out
        def minus(a):
            return (a - a.mean())/(a.std() + 1e-5)
        # standarlization along columns which is one specific neural activity trace for all time
        activity = np.apply_along_axis(minus, 1, Trajectory)
        cov = activity.T@activity
        # take eign vector and values
        self.vals, self.vect = np.linalg.eig(cov)
        
    def Dynamics(self, T_total = 200, T_stim = 100, T_duration = 60, Hiddens = [], noise = 2, iters = 4, 
Actions = [2, 0, 1, 3], e = 0,  same = True, legend = False, corner = False, open_loop = True, readout_random = False, h2o = 1) :
        self.Attractors = []
        self.Timescale = []
        self.Trajectory = []
        self.Positions = []
        self.PCs = np.zeros((5, 4, T_total))
        # take pca for specific game 
        self.Ts = []
        # like starting for different position 
        pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16)), ('empty', (5, 5))]
        colors = ['r', 'g', 'b', 'm', 'c']
        k = 0
        # control open loop or not , if open loop false, initialize a h2o with proper gain   
        if readout_random == True and open_loop == False:
            self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * h2o * np.sqrt(2.0/(512+4)))
        # all stimulu actions pairs by two loops 
        for iters1, pos in enumerate(pos_dict):
            for (iters2, action) in enumerate(Actions):
                pos0 = pos[1]
                Stim1 = T_stim * [torch.zeros(9)]
                Stim2 = T_duration * [torch.FloatTensor(self.game.grid.visible(pos0)).resize(9)]
                Stim3 = (T_total - (T_duration + T_stim)) * [torch.zeros(9)] 
                Stim = torch.stack(Stim1 + Stim2 + Stim3)
                # trace in empty room 
                Pos1, hidden1, dh1, actions = trajectory_empty(pos0, self.game, Stim, action_ = action, e = e, open_loop = open_loop)
                T_transient = np.sum(dh1[T_stim:]>1e-1)
                self.Ts.append(T_transient)
                self.PC_traces = self.vect[:5] @ hidden1.T
                self.PCs[iters1, iters2, :] = self.PC_traces[0, :].copy()
                # time threshold to assign limit cycle, take it for pwd
                if T_transient > T_total - T_stim  - 1:
                    self.Trajectory.append(self.PC_traces[0][100:])
                self.Positions.append(Pos1)
                # take the final value of h for mulitple stability
                self.Attractors.append(self.PC_traces[:5, -1])
        self.Attractors = np.array(self.Attractors)
        
    def Dynamics_2clicks(self, T_total = 200, T_stim1 = [20, 3], T_stim2 = [20, 3], 
                        Hiddens = [], noise = 2, iters = 4, 
action = [0], e = 0,  same = True, legend = False, corner = False, open_loop = True, readout_random = False, h2o = 1) :
        self.Attractors = []
        self.Timescale = []
        self.Trajectory = []
        self.Positions = []
        self.PCs = np.zeros((4, T_total))
        # take pca for specific game 
        self.Ts = []
        # like starting for different position 
        pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16))]
        k = 0
        # control open loop or not , if open loop false, initialize a h2o with proper gain   
        if readout_random == True and open_loop == False:
            self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * h2o * np.sqrt(2.0/(512+4)))
        # all stimulu actions pairs by two loops 
        for (iters1, pos1) in enumerate(pos_dict):
            pos1_ = pos1[1]
            pos2_ = pos1[1]
            Stim1 = T_stim1[0] * [torch.zeros(9)] + T_stim1[1] *[torch.FloatTensor(self.game.grid.visible(pos1_)).resize(9)]
            Stim2 = T_stim2[0] * [torch.zeros(9)] + T_stim2[1] *[torch.FloatTensor(self.game.grid.visible(pos2_)).resize(9)]
            Stim3 = (T_total - (T_stim1[0] + T_stim1[1] + T_stim2[0] + T_stim2[1])) * [torch.zeros(9)] 
            self.Stim = torch.stack(Stim1 + Stim2 + Stim3)
                # trace in empty room 
            Pos1, hidden1, dh1, actions = trajectory_empty(pos1_, self.game, self.Stim, action_ = action, e = e, open_loop = open_loop)
#                 T_transient = np.sum(dh1[T_stim:]>1e-1)
#                 self.Ts.append(T_transient)
            self.PC_traces = self.vect[:5] @ hidden1.T
            self.PCs[iters1, :] = self.PC_traces[0, :].copy()
                # time threshold to assign limit cycle, take it for pwd
#                 if T_transient > T_total - T_stim  - 1:
#                     self.Trajectory.append(self.PC_traces[0][100:])
            self.Positions.append(Pos1)
                # take the final value of h for mulitple stability
            self.Attractors.append(self.PC_traces[:5, -1])
        self.Attractors = np.array(self.Attractors)
        
## here we need to define a empty room, the network can inside the room with certian feedback given by Wih 
        
    def Dynamics_room(self, T_total = 200, Hiddens = [], iters = 4, e = 0,  readout_random = True, h2o = 1, starts = []):
            self.Attractors = []
            self.Timescale = []
            self.Trajectory = []
            self.Positions = []
            self.PCs = np.zeros((len(starts), T_total))
            # take pca for specific game 
            self.Ts = []
            # like starting for different position 
            pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16)), ('empty', (5, 5))]
            colors = ['r', 'g', 'b', 'm', 'c']
            k = 0
            # control open loop or not , if open loop false, initialize a h2o with proper gain   
            if readout_random == True:
                self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * h2o * np.sqrt(2.0/(512+4)))
            # all stimulu actions pairs by two loops 
            for iters, start in enumerate(starts):
                pos0 = start
                # trace in empty room 
                Pos1, hidden1, dh1, actions = trajectory_room(pos0, self.game, T_total = T_total, reward_control = 0, epsilon = 0, start = start)
#                 T_transient = np.sum(dh1[T_stim:]>1e-1)
#                 self.Ts.append(T_transient)
                self.PC_traces = self.vect[:5] @ hidden1.T
                self.PCs[iters, :] = self.PC_traces[0, :].copy()
                # time threshold to assign limit cycle, take it for pwd
#                 if T_transient > T_total - T_stim  - 1:
#                     self.Trajectory.append(self.PC_traces[0][100:])
                self.Positions.append(Pos1)
                # take the final value of h for mulitple stability
#                 self.Attractors.append(self.PC_traces[:5, -1])