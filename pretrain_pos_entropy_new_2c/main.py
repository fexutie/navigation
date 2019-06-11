# Pretraining with standard weigths scaling, lr rate function as 1e-5,  regulizor 1e-6.  The context is given as game, the loss function target is direct target without exp(-z2)


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

from collections import OrderedDict


import pretrain
from pretrain import *

import navigation2
from navigation2 import *

import Nets
from Nets import *

import warnings
warnings.filterwarnings('ignore')

# two different contexts, change beta in a curriculum way 
Pretest =  PretrainTest(weight_write = 'weights_cpu/rnn_1515tanh512_checkpoint', holes = 0, k_action =1, k_internal = 1)
for i in range(0, 400):
    if i<=10:
        k = 1
    else:
        k = (i//40 + 2) 
    beta = k * 1e-2
    Pretest.pretrain(i, pretrain = True, lr = 1e-5, beta = beta, beta_min = k * 1e-2, beta_max = k * 1e-2)
    net = Pretest.pregame.net.cpu()
    torch.save(net.state_dict(), 'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(i))
    
    