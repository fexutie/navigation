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

import navigation2
from navigation2 import *

import dynamics 
from dynamics import *

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
def Transform(States, Poss, Hiddens, Actions, Contexts, Sizes, size_range = np.arange(10, 51, 10)):
    Status = np.concatenate([State_transform(state, poss, size = size_range[int(size[0])]) for state, poss, size in zip(States, Poss, Sizes)])
    Hiddens = np.concatenate(Hiddens)
    Poss = np.concatenate(Poss)
    Actions = np.concatenate(Actions)
    # transform state to stim　
    States = np.concatenate(States)
    Contexts = np.concatenate(Contexts)
    return States, Poss, Hiddens, Actions, Status, Contexts

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

def State_transform(State, Poss, size = 15):
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

def Feature_preprocessing(States, Poss, Hiddens, Actions, Contexts, Sizes, size_max = 50):
    States, Poss, Hiddens, Actions, Status, Contexts = Transform(States, Poss, Hiddens, Actions, Contexts, Sizes)
    x =  Hiddens[:, :512]
    z = (x - np.min(x))/(np.max(x) - np.min(x))
    y = np.log(z/(1-z + 1e-3) + 1e-3)
    A = np.array([np.eye(4)[a] for a in Actions]).reshape(-1, 4)
    Y = np.array([np.eye(size_max)[y] for y in Poss[:,0] -2]).reshape(-1, size_max)
    X = np.array([np.eye(size_max)[x] for x in Poss[:,1] - 2]).reshape(-1, size_max)
    M = np.array([np.eye(5)[s] for s in Status]).reshape(-1, 5)
    S = States.reshape(-1, 9)
    C = np.array(Contexts).reshape(-1, 1)
    # S_wall = np.array([np.eye(27)[s] for s in Stim_wall]).reshape(-1, 27)
    Features = np.concatenate((A, Y, X, M, S, C), axis = 1)
    Features_A = np.concatenate((Y, X, M, S, C), axis = 1)
    Features_Y = np.concatenate((A, X, M, S, C), axis = 1)
    Features_X = np.concatenate((A, Y, M, S, C), axis = 1)
    Features_M = np.concatenate((A, Y, X, S, C), axis = 1)
    Features_S = np.concatenate((A, Y, X, M, C), axis = 1)
    Features_C = np.concatenate((A, Y, X, M, S), axis = 1)
    return y, Features, Features_A, Features_Y, Features_X, Features_M, Features_S, Features_C

# record sessions 100 for 2 different context, record the relevant variables 
def Data_record(weight, k_action = 1, epsilon = 0, size = 15, T = 200, seed_num = 1, input_type = 0):
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
        game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 0 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = input_type, discount = 0.9, alpha = 1
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

# record sessions 100 for 2 different context, record the relevant variables 
# consider size effect on samples ratio
def Data_record(weight, size_range = np.arange(10, 51, 10), T = 500, seed_num = 1e3, epsilon = 0):
    PC_traces = []
    Hiddens = []
    Poss = []
    Actions = []
    States = []
    Contexts = []
    Sizes = []
    for i in range(T):
        torch.manual_seed(np.random.randint(seed_num))
        hidden = torch.randn(1, 512)
        r = np.random.choice(len(size_range), 1, p = size_range[::-1]/np.sum(size_range))
        size = size_range[r[0]]
        start = (np.random.randint(2, size +2),  np.random.randint(2, size+2))
        game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 0 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = 0, discount = 0.9, alpha = 1
                           ,lam = 0)
        game.net.load_state_dict(torch.load(weight))
        
        grid = game.grid.grid.copy()
        c = np.random.randint(2)
        Pos, hidden, dh, Action, State, reward = trajectory(game, start, reward_control = c, size = size, epsilon = epsilon,\
                                                                  test = 0, limit_set = 4, init_hidden = False, hidden = hidden)
        Hiddens.append(hidden)
        Poss.append(Pos[1:])
        Actions.append(Action)
        States.append(State)
        Contexts.append(c * np.ones(len(State)))
        Sizes.append(r * np.ones(len(State)))
    return States, Poss, Hiddens, Actions, Contexts, Sizes 