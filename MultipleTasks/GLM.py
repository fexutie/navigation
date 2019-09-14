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

# import pretrain
# from pretrain import *

import navigation2
from navigation2 import *

import Nets 
from Nets import *

import sklearn
from sklearn.svm import SVC
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

import scipy
from scipy.spatial import distance
from scipy import signal


# Put the stimulus memory(status) and action base for memory and navigation  
def Transform(States, Poss, Hiddens, Actions, Context, history = False, size = 15):
    # last click state
    Status = np.concatenate([State_transform(state, poss, size) for state, poss in zip(States, Poss)])
    Hiddens = np.concatenate(Hiddens)
    Poss = np.concatenate(Poss)
    Actions = np.concatenate(Actions)
    Context = np.concatenate(Context)
    # transform state to stim　
    States = np.concatenate(States)
    # transform status to memory
    Memory = history_summary(Status)
    set_ = set([tuple(s) for s in States])
    dict_ = {}
    for i, s in enumerate(set_): 
        dict_.update({s:i})
    Stim = [dict_[tuple(s)] for s in States] 
    set_ = set([tuple(m) for m in Memory])
#     dict_ = {}
#     for i, m in enumerate(set_): 
#         dict_.update({m:i})
#     Mem = [dict_[tuple(m)] for m in Memory] 
    if history == False:
        return States, Poss, Hiddens, Actions, Status, Context
    else:
        return States, Poss, Hiddens, Actions, Status, Context, Memory 


# histroy memory of two 
def history_summary(Status):
#     S = [p[0]  for s, p in zip(State, Poss)]
    M = np.zeros((len(Status), 2))
    m2 = 0
    for i, (s1, s2) in enumerate(zip(Status[:-1], Status[1:])): 
        # if next clicks changes, then store this click as a memory value registed in m2  
        if s2 != s1:
            m2 = s1
        # sore memory,  m0 as memory of stimulus, m1 as memory of second click    
        M[i+1, 0] = s2
        M[i+1, 1] = m2   
#         print (s1)
    return M  
# State_transform()

# def Transform(States, Poss, Hiddens, Actions, Context):
#     Status = np.concatenate([State_transform(state, poss) for state, poss in zip(States, Poss)])
#     Hiddens = np.concatenate(Hiddens)
#     Poss = np.concatenate(Poss)
#     Actions = np.concatenate(Actions)
#     Context = np.concatenate(Context)
#     # transform state to stim　
#     States = np.concatenate(States)
#     return States, Poss, Hiddens, Actions, Status, Context

# transform to time section
def Stage_transform(State):
    S = np.cumsum([np.sum(s1 != s2) for (s1, s2) in zip(State[:-1], State[1:])] + [1]) 
    return S

# transform to space section
def wall_detection(pos, size):
    if pos[0] == 2:
        Stim = 1
    elif pos[0] == 2 + size - 1:
        Stim = 2
    elif pos[1] == 2:
        Stim = 3
    elif pos[1] == 2 + size - 1:
        Stim = 4
    else:
        Stim = 0 
    return Stim 
# for the memory feature, a Msimple one is just the last click, a most complicate one should be click sequence     
def State_transform(State, Poss, size):
#     S = [p[0]  for s, p in zip(State, Poss)]
    S = [wall_detection(pos, size) for pos in Poss]
    Status = []
    s1 = 0
    for s in S: 
        if s != 0: 
            s1 = s
#         print (s1)
        Status.append(s1)
    return Status
# histroy memory of two 
def history_summary(Status):
#     S = [p[0]  for s, p in zip(State, Poss)]
    M = np.zeros((len(Status), 2))
    m2 = 0
    for i, (s1, s2) in enumerate(zip(Status[:-1], Status[1:])): 
        # if next clicks changes, then store this click as a memory value registed in m2  
        if s2 != s1:
            m2 = s1
        # sore memory,  m0 as memory of stimulus, m1 as memory of second click    
        M[i+1, 0] = s2
        M[i+1, 1] = m2   
#         print (s1)
    return M

def Feature_preprocessing(States, Poss, Hiddens, Actions, Context, size = 15):
    States, Poss, Hiddens, Actions, Status, Context = Transform(States, Poss, Hiddens, Actions, Context)
    x =  Hiddens[:, :512]
    z = (x - np.min(x))/(np.max(x) - np.min(x))
    y = np.log(z/(1-z + 1e-3) + 1e-3)
    A = np.array([np.eye(4)[a] for a in Actions]).reshape(-1, 4)
    Y = np.array([np.eye(size)[y] for y in Poss[:,0] -2]).reshape(-1, size)
    X = np.array([np.eye(size)[x] for x in Poss[:,1] - 2]).reshape(-1, size)
    M = np.array([np.eye(5)[s] for s in Status]).reshape(-1, 5)
    S = States.reshape(-1, 9)
    C = np.array(Context).reshape(-1, 1)
    # S_wall = np.array([np.eye(27)[s] for s in Stim_wall]).reshape(-1, 27)
    Features = np.concatenate((A, Y, X, M, S, C), axis = 1)
    Features_A = np.concatenate((Y, X, M, S, C), axis = 1)
    Features_Y = np.concatenate((A, X, M, S, C), axis = 1)
    Features_X = np.concatenate((A, Y, M, S, C), axis = 1)
    Features_M = np.concatenate((A, Y, X, S, C), axis = 1)
    Features_S = np.concatenate((A, Y, X, M, C), axis = 1)
    Features_C = np.concatenate((A, Y, X, M, S), axis = 1)
    return y, Features, Features_A, Features_Y, Features_X, Features_M, Features_S, Features_C
# State_transform()

# record sessions 100 for 2 different context, record the relevant variables 
def Data_record(weight, k_action = 1, epsilon = 0, size = 15, T = 200, seed_num = 1e3):
    PC_traces = []
    Hiddens = []
    Poss = []
    Actions = []
    States = []
    Context = []
    for i in range(T):
        torch.manual_seed(np.random.randint(seed_num))
        hidden0 = torch.randn(1, 512)
        c  = np.random.randint(2)
        start = (np.random.randint(2, size +2),  np.random.randint(2, size+2))
        game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 0 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = 0, discount = 0.9, alpha = 1
                           ,lam = 0)
        game.net.load_state_dict(torch.load(weight))
        game.net.k_action = k_action 
        grid = game.grid.grid.copy()
        Pos, hidden, dh, Action, State, reward = trajectory(game, start, reward_control = c, size = size, \
                                                                  test = 0, limit_set = 4, init_hidden = False, hidden = hidden0, epsilon = epsilon)
        Hiddens.append(hidden)
        Poss.append(Pos[1:])
        Actions.append(Action)
        States.append(State)
        Context.append(c * np.ones(len(State)))
    return States, Poss, Hiddens, Actions, Context