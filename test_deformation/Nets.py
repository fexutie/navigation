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

import sklearn
from sklearn.svm import SVC

import scipy
from scipy.spatial import distance
from scipy import signal

import collections
from collections import OrderedDict


# A complete experiment including pretraining , decoding training, and q learning  
class PretrainTest():
    def __init__(self, weight_write, holes = 0, inputs_type = (1, 0)):
        self.pregame = PretrainGame(grid_size = (15, 15), holes = holes, random_seed = 4 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = inputs_type[0])
        self.game = ValueMaxGame(grid_size = (15, 15), holes = holes, random_seed = 4 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = inputs_type[1])
        self.weight = weight_write

    def matrix_shuffle(self, M):
        matrix = M.data.numpy()
        m, n = matrix.shape
        values = matrix.ravel()
        np.random.shuffle(values)
        matrix = values.reshape(m, n)
        return nn.Parameter(torch.FloatTensor(matrix))

    def matrix_randomize(self, M):
        matrix = M.data.numpy()
        m, n = matrix.shape
        sigma, mu = matrix.std(), matrix.mean()
        matrix = mu + sigma * torch.randn(m, n)
        return nn.Parameter(matrix)

    def weight_shuffle(self, h2h=True, ah=True, bh=True, ih=True):
        if h2h == True:
            self.game.net.h2h = self.matrix_shuffle(self.game.net.h2h)
        if ah == True:
            self.game.net.a2h = self.matrix_shuffle(self.game.net.a2h)
        if ih == True:
            self.game.net.i2h = self.matrix_shuffle(self.game.net.i2h)
        if bh == True:
            self.game.net.bh = self.matrix_shuffle(self.game.net.bh)

    def weight_randomnization(self, h2h=True, ah=True, bh=True, ih=True):
        if h2h == True:
            self.game.net.h2h = self.matrix_randomize(self.game.net.h2h)
        if ah == True:
            self.game.net.a2h = self.matrix_randomize(self.game.net.a2h)
        if ih == True:
            self.game.net.i2h = self.matrix_randomize(self.game.net.i2h)
        if bh == True:
            self.game.net.bh = self.matrix_randomize(self.game.net.bh)
            
    def loadweight(self, weight_load):
#       need to take the state dict as a new dict for updating 
        net_dict = torch.load(weight_load)
        list_modules = [('h2h', net_dict['h2h']), ('a2h', net_dict['a2h']), ('i2h', net_dict['i2h']), ('r2h', net_dict['r2h']), ('bh', net_dict['bh'])]
        select_dict = OrderedDict(list_modules)
        net = self.pregame.net.state_dict()
        net.update(select_dict)
        self.pregame.net.load_state_dict(net)
        torch.save(self.pregame.net.state_dict(), self.weight) 
    
        
    def pretrain(self, trial, weight = None, lr = 1e-5, pretrain = True):  
        # start a pretrained game  
        self.pregame.net.cuda()
        if pretrain == True:
            lr = float(lr)
            if weight != None:
                self.pregame.net.load_state_dict(torch.load(weight))
            self.pregame.fulltrain(lr_rate = lr, trials = int(1e3), batchsize = 4)
        print ('pretrain end', torch.norm(self.pregame.net.h2h))
        if pretrain == True:
            torch.save(self.pregame.net.state_dict(), self.weight[:-1]+'{}'.format(trial))
        else:
            torch.save(self.pregame.net.state_dict(), self.weight+'{}'.format(trial))
        if pretrain == True and trial <= 10:
            self.weight = self.weight[:-1]+'{}'.format(trial)
        elif pretrain == True and trial > 10:
            self.weight = self.weight[:-2]+'{}'.format(trial)
        elif pretrain == False:
            self.weight = self.weight +'{}'.format(trial)
            
              
    def decode(self, weight = None, size_range = [15], size_test = [15], epsilon = 0):
        if weight != None:
            self.game.net.load_state_dict(torch.load(weight))
        else:
            self.game.net.load_state_dict(torch.load(self.weight))
        rls_q = RLS(1e2)
        rls_sl = RLS(1e2)
        def precision(size = 15, reward_control = 0):
            dist, decode, visit, Cor_y, Cor_x = decodetest(self.game, reward_control= reward_control, epsilon = 0, size = size)
            prec0 = np.mean(dist)
            return np.mean(decode/visit), decode/visit
        Prec = 0
        # iterations are number of turns for decoder update, epochs are how many turns of games at each update  
        Prec_matrix = np.zeros((15, 15))
        for reward_control in [0, 1]:
            self.game.experiment(rls_q, rls_sl, iterations = 1, epochs = 100, epsilon = epsilon, train_hidden = False, train_q = False, size_range = size_range, test = True, decode = True, reward_control = reward_control) 
            prec, prec_matrix =  precision(size = size_test[0], reward_control = reward_control)
            Prec += prec
            Prec_matrix += prec_matrix
            print(self.game.reward_control, prec)
        # tested on size 15
        Prec = Prec/2
        Prec_matrix = Prec_matrix/2
        print ('decode train finish', Prec)
        return Prec, Prec_matrix

    
        
    def qlearn(self, weight_read, weight_write, iterations = 5, save = True, size_train = np.arange(10, 51, 10), \
               size_test = [10, 30], train_only = False, test_only = False, noise = 0.3, lam = 0.5, k_action = 1, h2o = True, shuffle = False):
        self.game.net.load_state_dict(torch.load(weight_read))
        if shuffle == True:
            self.weight_shuffle()
        if h2o == True:
            self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * 0.01 * np.sqrt(2.0/(512 + 4)))
        self.game.net.k_action = k_action    
        e_rate = [noise for r in range(iterations)] 
        rls_q = RLS(1e2, lam = 0.5)
        rls_sl = RLS(1e2)
        Rewards = []
        # q leanring phase
        for n,e in enumerate(e_rate):
            prob = np.ones(len(size_train)) 
            prob = prob/np.sum(prob)
#             self.game.seed_range = 2 + (n//10) * 10
            self.game.seed_range = 1e5
            if test_only == False:
                self.game.experiment(rls_q, rls_sl, iterations = 50, epochs= 10, epsilon = e, size_range = size_train)    
                if save == True:
                    torch.save(self.game.net.state_dict(), weight_write + '_{}'.format(n))
            def testing(game):
                Rewards00 = Test(game, reward_control = 0, size = size_test[0], test = 1)
                Rewards01 = Test(game, reward_control = 1, size = size_test[0], test = 1)
                rewards_s = (np.sum(Rewards00) + np.sum(Rewards01))/2
                if len(size_test) == 1:
                    return rewards_s
                else:
                    Rewards10 = Test(game, reward_control = 0, size = size_test[1], test = 2)
                    Rewards11 = Test(game, reward_control = 1, size = size_test[1], test = 2)
                    rewards_l = (np.sum(Rewards10) + np.sum(Rewards11))/2
                    return rewards_s, rewards_l
            # load weight if test only is true 
            if test_only == True:
                self.game.net.load_state_dict(torch.load(weight_write))
            if train_only == False:
                self.game.holes = 50
                rewards = testing(self.game)
            print (n, 'rewards',  rewards)
            Rewards.append(rewards)
            # adjust forget rate
            lam = np.sqrt((rewards + 1)/2)
            rls_q.lam = lam
        return Rewards
    
    def TestAllSizes(self, size_range = np.arange(15, 86, 10), limit_set = 8, test_size = None, grid = [], start = [], scale = 1):
        self.game.net.load_state_dict(torch.load(self.weight))
        self.Performance = []
        for size in size_range:
            if test_size == None: 
                test_size = size//25
            else:
                test_size = test_size
            Rewards0 = Test(self.game, reward_control = 0, size = size, test = test_size, limit_set = limit_set, map_set = grid, start = start, scale = scale)
            Rewards1 = Test(self.game, reward_control = 1, size = size, test = test_size, limit_set = limit_set, map_set = grid, start = start, scale = scale)
            self.Performance.append((Rewards0 + Rewards1)/2)




            
            
# attention to noise level, here corresponed to pretraining , so set noise to 1 
def trajectory(game, pos0, reward_control = 0, init_hidden = True, hidden = torch.zeros(1, 512), size = 19, test = 2, wind = (0, 0), map_set = [], limit_set = 32, epsilon = 0):
    game.reset(set_agent = pos0, reward_control = reward_control, size = size, limit_set = limit_set, test = test, train = False, map_set = map_set)
    done = False
    if init_hidden == True:
        game.hidden = game.net.initHidden()
    else:
        game.hidden = hidden.clone()
    hidden0 = game.hidden.clone()
    Hidden = []
    dH = []
    Action = []
    State = []
    Pos = []
    Pos.append(game.agent.pos)
    while not done:
        if (game.t >= wind[0] and game.t <= wind[1]):
            e = 1
        else: 
            e = 0
        pos0, state, reward, done = game.step(game.maxplay, epsilon = epsilon, test=True) # Down
        Hidden.append(game.hidden.data.numpy().squeeze())
        dH.append(torch.norm(game.hidden - hidden0))
        Pos.append(game.agent.pos)
        Action.append(np.argmax(game.action.data.numpy()))
        State.append(state)
        hidden0 = game.hidden.clone()
    return Pos, np.array(Hidden), np.array(dH), np.array(Action), np.array(State), reward

def trajectory_empty(pos0, game, T = 50, T0 = 100, Time_stop = 60, reward_control = 0, action_ = 0, e = 0, open_loop = True):
    game.reset(reward_control = reward_control, size = 15)
    done = False
    game.hidden = game.net.initHidden()
    game.action = torch.zeros(1, 4)
    Hidden = []
    Action = []
    Pos = []
    dH = []
    hidden0 = game.hidden.clone()
    game.agent.pos = pos0 
    stim = game.visible_state 
#     print (game.visible_state)
    y, x = 0, 0
    action = action_
    while not done:
        if open_loop == True:
            done, action = game.step_empty(stim = stim, action_ = action_, epsilon = e, T = T, T0 = T0, time_stop = Time_stop, open_loop = open_loop) # Down
        else:
            done, action = game.step_empty(stim = stim, action_ = action, epsilon = e, T = T, T0 = T0, time_stop = Time_stop, open_loop = open_loop) # Down
        # up 
        if action == 0: y -= 1
        # right
        elif action == 1: x += 1
        # down
        elif action == 2: y += 1
        # left
        elif action == 3: x -= 1
        game.agent.pos = (y, x)
        Pos.append(game.agent.pos)
        Hidden.append(game.hidden.clone().data.numpy().squeeze()) # need copy , avoid same adress 
        dH.append(torch.norm(game.hidden - hidden0))
        Action.append(action)
        hidden0 = game.hidden.clone()
    dH = [dh.data.numpy() for dh in dH]
    return Pos, np.array(Hidden), np.array(dH), np.array(Action)


## Principle components in real game and relaxation settings 
class PCA():
    def __init__(self, size = 15, reward_control = 0,  weight = None, net = 1, trial = 0, epsilon = 1, time_limit = 4):
        self.epsilon = epsilon
        self.reward_control = reward_control
        if weight == None:
            self.weight = 'weights_cpu{}/rnn_1515tanh512_checkpoint{}'.format(net, trial)
        else: 
            self.weight = weight
        self.game = ValueMaxGame(grid_size = (size, size), holes = 0, random_seed = 4 , set_reward = [(0.5, 0.25), (0.5, 0.75)], input_type = 0, action_control = 1, 
                                 discount = 0.9, alpha = 1, time_limit=100, lam = 0.5)
        self.game.net.load_state_dict(torch.load(self.weight))
        self.trial = trial
        self.size = size
        self.time_limit = time_limit 
    # record hidden activity, attention take test = true
    def record(self):
        self.Hiddens = np.zeros((self.size * self.size, self.time_limit * self.size, 512))
        for j in range (2,self.size + 2):
            for i in range (2, self.size + 2):
                self.game.reset(set_agent = (j,i), reward_control = self.reward_control, size = self.size)
                done = False
                self.game.hidden = self.game.net.initHidden()
                for t in range(self.time_limit * self.size):
                    pos0, state, reward, done = self.game.step(self.game.maxplay, epsilon = self.epsilon, test = True) # Down
                    self.Hiddens[(j-2) * self.size + (i-2), t] = (self.game.hidden.data.numpy().squeeze())
                    if done == True:
                        self.game.reset(reward_control = self.reward_control, size = self.size)
                        done = False 
    
    def record_empty(self):
        pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16)), ('upleft',(2,2)), \
                    ('upright',(2,16)), ('downleft', (16,2)), ('downright', (16,16)), ('reward', (9,12)), ('empty', (5, 5))]
        self.size_dict = len(pos_dict)
        self.Hiddens = np.zeros((4 * len(pos_dict), 401, 512))
        k = 0
        # all stimulus for all actions 
        Actions = [2, 0, 1, 3]
        for iters, pos in enumerate(pos_dict):
            for act, action in enumerate(Actions):
                pos0 = pos[1]
                self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * 0.01 * np.sqrt(2.0/(512+4)))
                Pos1, hidden1, dh1, actions = trajectory_empty(pos0, self.game, T = 400, T0 = 200, action_ = action, \
                                                        e = 0, Time_stop = 3, open_loop = False)
                self.Hiddens[iters * len(Actions) + act] =  hidden1 
    # put every row(time) together into one long trajectory, which means concatenation along rows - number of neurons
    def pca(self):
        self.record()
        Trajectory = self.Hiddens.reshape(self.size * self.size * self.time_limit * self.size, 512)
        #  take correlation out
        def minus(a):
            return (a - a.mean())/(a.std() + 1e-5)
        # standarlization along columns which is one specific neural activity trace for all time
        activity = np.apply_along_axis(minus, 1, Trajectory)
        cov = activity.T@activity
        # take eign vector and values
        self.vals, self.vect = np.linalg.eig(cov)

# Check the PCA and timescales 
    def TimeScale(self, plot = False, T_total = 800, Hiddens = [], noise = 2, iters = 4, Actions = [2, 0, 1, 3], e = 0, T0 = 100, Time_stop = 60, same = True, legend = False, corner = False, open_loop = True, readout_random = False, h2o = 1) :
        Attractors = []
        Timescale = []
        Trajectory = []
        Positions = []
        # take pca for specific game 
        Ts = []
        pos_dict = [('up',(2,9)), ('down', (16,9)), ('left', (9,2)), ('right',(9,16)), ('upleft',(2,2)), ('upright',(2,16)), ('downleft', (16,2)), ('downright', (16,16)), ('reward', (9,12)), ('empty', (5, 5))]
        colors = ['r', 'g', 'b', 'm', 'c']
        k = 0
        if readout_random == True and open_loop == False:
            self.game.net.h2o = nn.Parameter(torch.randn(512, 4) * h2o * np.sqrt(2.0/(512+4)))
        # all stimulus for specific actions 
        for iters, pos in enumerate(pos_dict):
            for action in Actions:
                pos0 = pos[1]
                Pos1, hidden1, dh1, actions = trajectory_empty(pos0, self.game, T = T_total, T0 = T0, action_ = action, e = e, Time_stop = Time_stop, open_loop = open_loop)
                Ts.append(np.sum((dh1[T0:]>1e-1)))
        #         print ('timescale', np.sum(dh1>1))
                PC_traces = self.vect[:10] @ hidden1.T
                if Ts[-1]>300:
                    Trajectory.append(PC_traces[0][100:])
                Positions.append(Pos1[100:])
#                 Hiddens.append((hidden1, pos[0]))
#                 if np.sum((dh1[T0:]>1e-1))<300:
                Attractors.append(PC_traces[:5, -1])
                if plot == True:
                    if same == False:
                        plt.figure(k)
                    else:
                        plt.figure(0)
                    if corner == False:
                        if (pos[0] == 'up') or (pos[0] == 'down') or (pos[0] == 'left') or (pos[0] == 'right') or (pos[0] == 'empty'):
                            plt.plot(PC_traces[0], label = pos[0] + str(self.trial), color = colors[k], alpha = 0.5 + 0.1 * k)
                            k += 1
                            if legend == True:
                                plt.legend()
                    else:
                        if (pos[0] == 'upleft') or (pos[0] == 'downleft') or (pos[0] == 'upright') or (pos[0] == 'downright') or (pos[0] == 'empty'):
                            plt.plot(PC_traces[0], label = pos[0] + str(self.trial), color = colors[k])
                            k += 1
                            if legend == True:
                                plt.legend()
        # check the distribution of ts for different stimulus and get out the mean and variance of it.  
        Attractors = np.array(Attractors)
        return Ts, Hiddens, Trajectory, Positions, np.mean(distance.cdist(Attractors, Attractors, 'euclidean'))
    
## Check receptive field, pay attention to time span and noise level, the shuffling process is added
class ActivityCheck():
    def __init__(self, size = 15, num_pc = 3):
        self.Grid = np.zeros((512, size + 4, size + 4))
        self.Grid_pc = np.zeros((num_pc, size + 4, size + 4))
        self.Grid_shuffle = np.zeros((512, size + 4, size + 4))
        self.Visit = 1e-5 * np.ones((size + 4, size + 4))
        self.Visit_shuffle = 1e-5 * np.ones((size + 4, size + 4))
        self.size = size
        self.num_pc = num_pc
    def activity(self, reward_control = 0, weight = 0, pc = False, vects = None, noise = 0.1):    
        game = ValueMaxGame(grid_size = (self.size, self.size), holes = 0, random_seed = 4 , set_reward = [], input_type = 0, action_control = 1, discount = 0.9, alpha = 1, time_limit = 8 * self.size, lam = 0.)
        game.net.load_state_dict(torch.load(weight))
        if reward_control == 0:
            context = (0.5, 0.25)
        else:
            context = (0.5, 0.75)
        for i in range (2, self.size + 2):
            for j in range (2, self.size + 2):
                game.reset(set_agent = (j,i), reward_control = reward_control, size = self.size, context = context, limit_set = 8 * self.size)
                done = False
                game.hidden = game.net.initHidden()
                np.random.seed()
                while not done:
                    pos0, state, reward, done = game.step(game.maxplay, epsilon = noise, record=True, test = True) # Down
                    self.Grid[:, game.agent.pos[0], game.agent.pos[1]] +=  np.abs(game.hidden[0].cpu().data.numpy())
                    x, y = np.random.randint(2, self.size + 2), np.random.randint(2, self.size + 2)
                    self.Grid_shuffle[:, x, y] +=  np.abs(game.hidden[0].cpu().data.numpy())
                    if pc == True:
                            self.Grid_pc[:, game.agent.pos[0], game.agent.pos[1]] +=  np.abs(vects[:self.num_pc] @ game.hidden[0].cpu().data.numpy())
#                     self.Grid_shuffle[:, game.agent.pos[0], game.agent.pos[1]] +=  np.abs(game.hidden[0].cpu().data.numpy())
                    self.Visit[game.agent.pos] += 1
                    self.Visit_shuffle[x, y] += 1