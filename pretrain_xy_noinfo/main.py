# lr 1e-5, decay 1e-5 start from kah = 4 khh = 1.2 hih = 1, short pretrain, info on wall

from torch.autograd import Variable
import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch.nn import init
from torch.nn import DataParallel
from torch.utils.data import DataLoader


import seaborn as sns
from IPython.display import HTML

import pretrain
from pretrain import *

import navigation2
from navigation2 import *

import sklearn
from sklearn.svm import SVC

import scipy
from scipy.spatial import distance
from scipy import signal

import Nets
from Nets import*

print ('start')
Loss = []
Pretest =  PretrainTest(weight_write = 'weights_cpu/rnn_1515tanh512_checkpoint', holes = 0, k_action = 4, k_internal = 1.2, k_stim =1)

for i in range(40):
    Pretest.pretrain(i, pretrain = (i!=0), lr = 1e-5, decay = 1e-5, Loss = Loss)
    np.save('Loss', Loss)
    net = Pretest.pregame.net.cpu()
    torch.save(net.state_dict(), 'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(i))
print ('end')
