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

import Tests
from Tests import*


# moving bar train
trial = 399
tasks = ['scale']
iterations = [4]
# iterations = [1, 1, 1, 1, 1, 1, 1]
Scores = np.zeros((1, 5))
i = 0
for iters, task in zip(iterations, tasks):
        Task = MultipleTasks(task=task, weight_write='weights_cpu/rnn_1515tanh512_checkpoint{}'.format(0) \
                         , noise=0.0, weight1='weights_cpu_pos/rnn_1515tanh512_checkpoint399',
                         weight2='weights_cpu_mem/rnn_1515tanh512_checkpoint49')
        for k in range(1):
            weight_read = 'weights_' + task + '/rnn_1515tanh512_checkpoint{}_{}'.format(trial, 4 - k)
            if task == 'scale':
                score = Test(task, Task.game, weight = weight_read, size = 50, test_size = 2, limit_set = 4)
            elif (task == 'scale_x') or (task == 'scale_y'):
                score = Test(task, Task.game, weight = weight_read, test_size = 1, limit_set = 4)
            else:
                score = Test(task, Task.game, weight = weight_read, limit_set = 2)
            Scores[i, k]= score
        i += 1
np.save('Scores', Scores)