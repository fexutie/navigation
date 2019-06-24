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
from IPython.display import HTML

from collections import OrderedDict


import pretrain
from pretrain import *

import navigation2
from navigation2 import *

import Nets
from Nets import*

import warnings
warnings.filterwarnings('ignore')

Pretest =  PretrainTest(weight_write = 'weights_cpu/rnn_1515tanh512_checkpoint', holes = 0)

for i in range(400):
    Pretest.pretrain(i, pretrain = (i!=0), lr = 1e-5)
    net = Pretest.pregame.net.cpu()
    torch.save(net.state_dict(), 'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(i))
    
    
    