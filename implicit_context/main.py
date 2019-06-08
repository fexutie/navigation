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

import Nets
from Nets import*

import navigation2
from navigation2 import *

%pylab inline
import warnings


for iters, noise in enumerate(3 * [0.0]):
    for trial in [39]: 
        Pretest =  PretrainTest(holes = 0, weight_write = '/home/tie/Research/PhD/NavigationPaper_606/pretrain/pretrain_pos_origin2/weights_cpu/rnn_1515tanh512_checkpoint{}'.format(trial))
        weight_read = Pretest.weight
        weight_write = 'weights2/rnn_1515tanh512_checkpoint{}_{}'.format(trial, iters)
        rewards = Pretest.qlearn(weight_read,  weight_write, iterations = 30, noise = noise, size_train =[15], size_test=[15])
        np.save('Rewards_{}_{}.npy'.format(iters, trial), rewards)