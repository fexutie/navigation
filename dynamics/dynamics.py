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
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    return Pos, np.array(Hidden), np.array(dH), np.array(Action), np.array(State), reward

def trajectory_empty(pos0, game, Stim, reward_control = 0, action_ = [], e = 0, open_loop = True, init_hidden = True, hidden = torch.zeros(512, 512), context = torch.zeros(1, 38)):
    game.reset(reward_control = reward_control, size = 15)
    done = False
    if init_hidden == True:
        game.hidden = game.net.initHidden()
    else:
        game.hidden = hidden.clone()
    game.action = torch.zeros(1, 4)
    Hidden = []
    Action = []
    Y = []
    X = []
    dH = []
    hidden0 = game.hidden.clone()
    game.agent.pos = pos0 
    stim = game.visible_state 
#     print (game.visible_state)
    y, x = 0, 0
    action = action_
    for stim in Stim:
        if open_loop == True:
            action = game.step_empty(stim = stim, action_ = action_, epsilon = e, open_loop = open_loop, context = context) # Down
        else:
            action = game.step_empty(stim = stim, action_ = action, epsilon = e, open_loop = open_loop, context = context) # Down
        # up 
        if action == 0: y -= 1
        # right
        elif action == 1: x += 1
        # down
        elif action == 2: y += 1
        # left
        elif action == 3: x -= 1
        game.agent.pos = (y, x)
        Y.append(game.agent.pos[0])
        X.append(game.agent.pos[1])
        Hidden.append(game.hidden.clone().data.numpy().squeeze()) # need copy , avoid same adress 
        dH.append(torch.norm(game.hidden - hidden0))
        Action.append(action)
        hidden0 = game.hidden.clone()
    dH = [dh.data.numpy() for dh in dH]
    return (np.array(Y), np.array(X)), np.array(Hidden), np.array(dH), np.array(Action)


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
    def __init__(self, weight, size = 15, reward_control = 0, epsilon = 1):
        self.epsilon = epsilon
        self.reward_control = reward_control
        self.weight = weight
        self.game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 4 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = 0, action_control = 1, 
                                 discount = 0.9, alpha = 1, time_limit=100, lam = 0.5)
        self.game.net.load_state_dict(torch.load(self.weight))
    # record hidden activity, attention take test = true
    def record(self, T_duration = 50):
        self.Hiddens = np.zeros((225, 50, 512))
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
Actions = [2, 0, 1, 3], e = 0,  same = True, legend = False, corner = False, open_loop = True, readout_random = False, h2o = 1, context = torch.zeros(1, 38)) :
        self.Attractors = []
        self.Timescale = []
        self.Trajectory = []
        self.Ys = np.zeros((4, len(Actions), T_total - T_stim))
        self.Xs = np.zeros((4, len(Actions), T_total - T_stim))
        self.Actions = np.zeros((4, len(Actions),  T_total - T_stim))
        self.PCs = np.zeros((4, len(Actions), T_total))
        self.Hiddens = np.zeros((4, len(Actions), T_total - T_stim, 512))
        # take pca for specific game 
        self.Ts = []
        # like starting for different position 
        pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16))]
        colors = ['r', 'g', 'b', 'm', 'c']
        k = 0
        # control open loop or not , if open loop false, initialize a h2o with proper gain   
        if readout_random == True and open_loop == False:
            self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * h2o * np.sqrt(2.0/(512+4)))
        # all stimulu actions pairs by two loops 
        for iters1, pos in enumerate(pos_dict):
            pos0 = pos[1]
            Stim1 = T_stim * [torch.zeros(9)]
            Stim2 = T_duration * [torch.FloatTensor(self.game.grid.visible(pos0)).resize(9)]
            Stim3 = (T_total - (T_duration + T_stim)) * [torch.zeros(9)] 
            self.Stim = torch.stack(Stim1 + Stim2 + Stim3).view(-1, 9)
            for (iters2, action) in enumerate(Actions):

                # trace in empty room 
                Pos1, hidden1, dh1, actions = trajectory_empty(pos0, self.game, self.Stim, action_ = action, e = e, open_loop = open_loop, context = context)
                T_transient = np.sum(dh1[T_stim:]>1e-1)
                self.Ts.append(T_transient)
                self.PC_traces = self.vect[:5] @ hidden1.T
                self.PCs[iters1, iters2, :] = self.PC_traces[0, :].copy()
                # record the hidden activity after stimulus 
                self.Hiddens[iters1, iters2, :, :] = hidden1[T_stim:, :]
                # time threshold to assign limit cycle, take it for pwd
                if T_transient > T_total - T_stim  - 1:
                    self.Trajectory.append(self.PC_traces[0][100:])
                self.Ys[iters1, iters2] = Pos1[0][T_stim:]
                self.Xs[iters1, iters2] = Pos1[1][T_stim:]
                self.Actions[iters1, iters2] = actions[T_stim:]
        
    def Dynamics_2clicks(self, T_total = 200, T_stim1 = [20, 3], T_stim2 = [20, 3], wall2 = -1,
                        Hiddens = [], noise = 2, iters = 4, context = torch.zeros(1, 38),
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
            if wall2 == -1:
                pos2_ = pos1[1]
            else:
                pos2_ = pos_dict[wall2][1]
            Stim1 = T_stim1[0] * [torch.zeros(9)] + T_stim1[1] *[torch.FloatTensor(self.game.grid.visible(pos1_)).resize(9)]
            Stim2 = T_stim2[0] * [torch.zeros(9)] + T_stim2[1] *[torch.FloatTensor(self.game.grid.visible(pos2_)).resize(9)]
            Stim3 = (T_total - (T_stim1[0] + T_stim1[1] + T_stim2[0] + T_stim2[1])) * [torch.zeros(9)] 
            self.Stim = torch.stack(Stim1 + Stim2 + Stim3)
                # trace in empty room 
            Pos1, hidden1, dh1, actions = trajectory_empty(pos1_, self.game, self.Stim, action_ = action, e = e, open_loop = open_loop, context = context)
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

def variance_decompose(pca, T1, T2, open_loop = True, alpha = 1e-4):
    Actions = np.zeros((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T))
    Y = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T))
    X = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T))
    Stims = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T))
    colors = ['b', 'g', 'r', 'm']
    for i in range(pca.Hiddens.shape[0]):
        for j in range(pca.Hiddens.shape[1]):
            if open_loop == False:
                Actions[i, j] = pca.Actions[i, j, :T]
            else:
                Actions[i, j] = j
            Stims[i, j] = i
            Y[i, j] = pca.Ys[i, j, :T]
            X[i, j] = pca.Xs[i, j, :T]

    x =  pca.Hiddens[:, :, :T, :].reshape(-1, 512)
    z = (x - np.min(x))/(np.max(x) - np.min(x))
    y = np.log(z/(1-z + 1e-3) + 1e-3)
     ## features:
    A = np.array([np.eye(4)[int(a)] for a in Actions.reshape(-1)]).reshape(-1, 4)
    S = np.array([np.eye(4)[int(s)] for s in Stims.reshape(-1)]).reshape(-1, 4)
    Y = Y.reshape(-1, 1)/np.max(np.abs(Y))
    X = X.reshape(-1, 1)/np.max(np.abs(X))
    Features = np.concatenate((A, S, Y, X), axis = 1)
    Features_A = np.concatenate((S, Y, X), axis = 1)
    Features_S = np.concatenate((A, Y, X), axis = 1)
    Features_Y = np.concatenate((A, S, X), axis = 1)
    Features_X = np.concatenate((A, S, Y), axis = 1)

    clf1 = Lasso(alpha = alpha)
    clf1.fit(Features, y)
    y_pred = clf1.predict(Features)
    Importances = []
    importances = []
    for feature in [Features_A, Features_S, Features_Y, Features_X]:
        clf = Lasso(alpha= alpha)
        clf.fit(feature, y)
        y_pred_i = clf.predict(feature)
#         print (r2_score(y, y_pred_i), r2_score(y, y_pred))
        importances.append(r2_score(y, y_pred) - r2_score(y, y_pred_i))
        Importance = []
        for i in range(512):
            dif =  r2_score(y[:, i], y_pred[:, i]) - r2_score(y[:, i], y_pred_i[:, i])
            if dif<0:
                dif = 0
            Importance.append(dif)
        Importances.append(Importance)
    return importances

def Memory(weight, k_action = 1, k_stim = 1, k_internal = 1, epsilon = 1, context_gain = 1):
    # reference from net 1 
    def placefield(pos): 
        field =np.zeros((2, 19))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                pos_relative = pos[k]
                field[k, i] =  (i- pos_relative) ** 2 
        field = - 0.1 * torch.from_numpy(field).resize(1, 2 * 19).float()
        return field
    Info_P = np.zeros(8)
    Info_I = np.zeros(8)
    Info_A = np.zeros(8)
    pca = PCA(weight = weight)
    pca.pca(T_duration = 3)
    pca.game.net.k_action = k_action
    pca.game.net.k_internal = k_internal
    pca.game.net.k_stim = k_stim
    for pos_reward in [(9, 5), (9, 13)]:
        context = context_gain * placefield(pos_reward)
        pca.Dynamics(Actions = 10 * [0], legend = True, T_total = 200, T_stim = 30, T_duration = 3, \
                     readout_random = True, open_loop = False, e = epsilon, context = context)
        for i in range(8):
            T = 20 + i * 20
            importances = variance_decompose(pca, T, open_loop=False)
            Info_A[i] += importances[0]
            Info_I[i] += importances[1]
            Info_P[i] += importances[2] + importances[3]  
    Info_A = Info_A/2    
    Info_P = Info_P/2  
    Info_I = Info_I/2  
    print (np.mean(Info_A), np.mean(Info_I), np.mean(Info_P))
    return Info_A, Info_I, Info_P 

def variance_decompose(pca, T1, T2, open_loop = True, alpha = 1e-4):
    Actions = np.zeros((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T2-T1))
    Y = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T2-T1))
    X = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T2-T1))
    Stims = np.ones((pca.Hiddens.shape[0], pca.Hiddens.shape[1], T2-T1))
    colors = ['b', 'g', 'r', 'm']
    for i in range(pca.Hiddens.shape[0]):
        for j in range(pca.Hiddens.shape[1]):
            if open_loop == False:
                Actions[i, j] = pca.Actions[i, j, T1:T2]
            else:
                Actions[i, j] = j
            Stims[i, j] = i
            Y[i, j] = pca.Ys[i, j, T1:T2]
            X[i, j] = pca.Xs[i, j, T1:T2]

    x =  pca.Hiddens[:, :, T1:T2, :].reshape(-1, 512)
    z = (x - np.min(x))/(np.max(x) - np.min(x))
    y = np.log(z/(1-z + 1e-3) + 1e-3)
     ## features:
    A = np.array([np.eye(4)[int(a)] for a in Actions.reshape(-1)]).reshape(-1, 4)
    S = np.array([np.eye(4)[int(s)] for s in Stims.reshape(-1)]).reshape(-1, 4)
    Y = Y.reshape(-1, 1)/np.max(np.abs(Y))
    X = X.reshape(-1, 1)/np.max(np.abs(X))
    Features = np.concatenate((A, S, Y, X), axis = 1)
    Features_A = np.concatenate((S, Y, X), axis = 1)
    Features_S = np.concatenate((A, Y, X), axis = 1)
    Features_Y = np.concatenate((A, S, X), axis = 1)
    Features_X = np.concatenate((A, S, Y), axis = 1)

    clf1 = Lasso(alpha = alpha)
    clf1.fit(Features, y)
    y_pred = clf1.predict(Features)
    Importances = []
    importances = []
    for feature in [Features_A, Features_S, Features_Y, Features_X]:
        clf = Lasso(alpha= alpha)
        clf.fit(feature, y)
        y_pred_i = clf.predict(feature)
#         print (r2_score(y, y_pred_i), r2_score(y, y_pred))
        importances.append(r2_score(y, y_pred) - r2_score(y, y_pred_i))
        Importance = []
        for i in range(512):
            dif =  r2_score(y[:, i], y_pred[:, i]) - r2_score(y[:, i], y_pred_i[:, i])
            if dif<0:
                dif = 0
            Importance.append(dif)
        Importances.append(Importance)
    return importances

def Memory(weight, k_action = 1, k_stim = 1, k_internal = 1, epsilon = 1, context_gain = 1):
    # reference from net 1 
    def placefield(pos): 
        field =np.zeros((2, 19))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                pos_relative = pos[k]
                field[k, i] =  (i- pos_relative) ** 2 
        field = - 0.1 * torch.from_numpy(field).resize(1, 2 * 19).float()
        return field
    Info_P = np.zeros(8)
    Info_I = np.zeros(8)
    Info_A = np.zeros(8)
    pca = PCA(weight = weight)
    pca.pca(T_duration = 3)
    pca.game.net.k_action = k_action
    pca.game.net.k_internal = k_internal
    pca.game.net.k_stim = k_stim
    for pos_reward in [(9, 5), (9, 13)]:
        context = context_gain * placefield(pos_reward)
        pca.Dynamics(Actions = 100 * [0], legend = True, T_total = 200, T_stim = 30, T_duration = 3, \
                     readout_random = True, open_loop = False, e = epsilon, context = context)
        for i in range(8):
            T1 = i * 20
            T2 = 20 + i * 20
            importances = variance_decompose(pca, T1, T2, open_loop=False)
            Info_A[i] += importances[0]
            Info_I[i] += importances[1]
            Info_P[i] += importances[2] + importances[3]  
    Info_A = Info_A/2    
    Info_P = Info_P/2  
    Info_I = Info_I/2  
    print (np.mean(Info_A), np.mean(Info_I), np.mean(Info_P))
    return Info_A, Info_I, Info_P 