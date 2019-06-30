%load_ext autoreload
%autoreload 2

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
from sklearn.model_selection import cross_validate

import scipy
from scipy.spatial import distance
from scipy import signal

import dynamics 
from dynamics import * 


%pylab inline
import warnings
warnings.filterwarnings('ignore')


#Build a classifier for wall deocding of three networks 
# echo
weight ='weights_cpu/rnn_1515tanh512_checkpoint0'

# record sessions 100 for 2 different context, record the relevant variables 
def Data_record(weight, k_action = 1, epsilon = 0, size = 15, T = 500, seed_num = 1e3):
    PC_traces = []
    Hiddens = []
    Poss = []
    Actions = []
    States = []
    Context = []
    Values = []
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
        Pos, hidden, dh, Action, State, values, reward = trajectory(game, start, reward_control = c, size = size, \
                                                                  test = 0, limit_set = 8, init_hidden = False, hidden = hidden0, epsilon = epsilon, reward = True)
        Hiddens.append(hidden)
        Poss.append(Pos[1:])
        Actions.append(Action)
        States.append(State)
        Context.append(c * np.ones(len(State)))
        Values.append(values)
    return States, Poss, Hiddens, Actions, Context, Values

def Transform(States, Poss, Hiddens, Actions, Context, Values, history = False, size = 15):
    # last click state
    Borders = np.concatenate([State_transform(state, poss, size)[0] for state, poss in zip(States, Poss)])
    Status = np.concatenate([State_transform(state, poss, size)[1] for state, poss in zip(States, Poss)])
    Hiddens = np.concatenate(Hiddens)
    Poss = np.concatenate(Poss)
    Actions = np.concatenate(Actions)
    Context = np.concatenate(Context)
    # transform state to stim　
    States = np.concatenate(States)
    Values = np.concatenate(Values)
    # transform status to memory
    return Borders[Status>0], Poss[Status>0], Hiddens[Status>0], Actions[Status>0], Status[Status>0], Context[Status>0], Values[Status>0]

States, Poss, Hiddens, Actions, Context = Data_record(weight, k_action = 1, T = 500, seed_num = 1e6, epsilon = 1, size = 15)
States, Poss, Hiddens, Actions, Status, Context = Transform(States, Poss, Hiddens, Actions, Context)
X = np.array(Hiddens[Status>1])
Y = Status[Status>1]
score0 = cross_validate(model, X,Y, cv = 5)
# Pos
weight ='weights_cpu/rnn_1515tanh512_checkpoint300'
States, Poss, Hiddens, Actions, Context = Data_record(weight, k_action = 1, T = 500, seed_num = 1e6, epsilon = 1, size = 15)
States, Poss, Hiddens, Actions, Status, Context = Transform(States, Poss, Hiddens, Actions, Context)
X = np.array(Hiddens[Status>1])
Y = Status[Status>1]
score_pos = cross_validate(model, X,Y, cv = 5)
# Mem
weight ='weights_cpu_pre2/rnn_1515tanh512_checkpoint0'
Pretest = PretrainTest(weight, holes = 0, inputs_type = (0, 0))
Pretest.loadweight(weight)
States, Poss, Hiddens, Actions, Context = Data_record(weight, k_action = 1, T = 500, seed_num = 1e6, epsilon = 1, size = 15)
States, Poss, Hiddens, Actions, Status, Context = Transform(States, Poss, Hiddens, Actions, Context)
X = np.array(Hiddens[Status>1])
Y = Status[Status>1]
score = cross_validate(model, X,Y, cv = 5)
