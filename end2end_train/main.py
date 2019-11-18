# moving bar train, noise = 0.0

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

import Nets
from Nets import*


trial = 0
tasks = ['basic']
episodes = [200]
# iterations = [1, 1, 1, 1, 1, 1, 1]
for n, task in zip(episodes, tasks):
        Task =  MultipleTasks(task = task, weight_write = 'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(trial)\
                              , noise = 0.0)
        weight_read = Task.weight
        weight_write = 'weights_' + task + '/rnn_1515tanh512_checkpoint{}'.format(trial)
        if task == 'scale':
            size_train = np.arange(10, 51, 10)
        else:
            size_train = [15]
        if task == 'bar':
            iterations = 5
            epochs = 100
        else:
            iterations = 50
            epochs = 10
        print ('start', size_train)
        Task.qlearn(task, weight_read,  weight_write, episodes = n, noise = 0, size_train = size_train, size_test=[15], iterations = iterations, epochs = epochs, sgd = True)