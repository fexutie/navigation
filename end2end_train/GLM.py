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

import navigation2
from navigation2 import *

import Nets 
from Nets import *

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
    S = [0] + list(0.5 * np.cumsum([np.any((s1 != s2) & (np.sum(s1)<=0) & (np.sum(s2)<=0)) for (s1, s2) in zip(State[:-1], State[1:])])) 
    return np.array(S)

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
    M = [wall_detection(pos, size) for pos in Poss]
    Status = []
    m1 = 0
    for s, m in zip(State, M): 
        if m != 0: 
            m1 = m
        if np.sum(s)>0:
            m1 = -1
#         print (s1)
        Status.append(m1)
    return Status
# histroy memory of two 
