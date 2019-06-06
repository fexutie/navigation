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

import Nets
from Nets import*

import navigation2
from navigation2 import *

def Strategy_count(weight, size = 15, T = 5000, reward_control = 0, init_hidden = True):
    Hiddens = []
    Poss = []
    Actions = []
    States = []
    Pretest =  PretrainTest(holes = 0, weight_write = weight, inputs_type=(1, 0))   
    Pretest.game.net.load_state_dict(torch.load(weight))
    Events_freq = []
    Events = []
    torch.manual_seed(0)
    hidden0 = torch.randn(1, 512)
    for i in range(T):
        start = (np.random.randint(2, size +2),  np.random.randint(2, size+2))
    #     start = (5, 4)
        Pos, hidden, dh, Action, State, reward = trajectory(Pretest.game, start, reward_control = reward_control, size = size, test = 0, \
                                                                            init_hidden = init_hidden, hidden = hidden0, animate = False)
        Hiddens.append(hidden)
        Poss.append(Pos[1:])
        Actions.append(Action)
        States.append(State)
        #count clicks
        stim_index = np.array([np.any(s1 != s2) for s1, s2 in zip(State[:-1], State[1:])])
        stim_index = [False if (s1 == True) else s2 for s1, s2 in zip (stim_index[:-1], stim_index[1:])]
        events = np.array(Pos[1:-2])[stim_index]
        Events_freq.append(len(events)-1)
        Events.append(events)
    return Events, Events_freq, Poss

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

def dict_transform(Poss, size = 15):
#     S = [p[0]  for s, p in zip(State, Poss)]
    S = [wall_detection(pos, size) for pos in Poss]
    Status = []
    s0 = 0
    for s in S: 
        if (s != 0):
            if s0!= s:
                Status.append(s)
            s0 = s
    return tuple(Status)

def dictionary_plot(Poss):
    # State_transform()
    event_trace = [dict_transform(poss) for poss in Poss]
    # select 2 clicks 
    event_trace_select = list(filter(lambda x: len(x) == 2, event_trace))
    # form dictionary of visis 
#     set_ = set(tuple(event) for event in event_trace_select)
    set_ = [(i, j) for i in range(1, 5) for j in range(1, 5)]
    Dict_ = {} 
    for s in set_:
        num = event_trace_select.count((s[0], s[1]))
        Dict_.update([(s, num)] )
    Dict_real = {1: 'N', 2: 'S', 3: 'E', 4: 'W'}
    Dict_final = {}
    for word in list(Dict_.keys()):
        Dict_final.update({(Dict_real[word[0]] + Dict_real[word[1]]): Dict_[word]})
    plt.bar(range(len(Dict_)), list(Dict_.values()), align='center')
    plt.xticks(range(len(Dict_)), list(Dict_final.keys()))
    freq = np.array(list(Dict_.values()))
    print (freq)
    p = freq/np.sum(freq)
    entropy = -np.sum(p * np.log(p + 1e-5))
    print (entropy)
    return entropy